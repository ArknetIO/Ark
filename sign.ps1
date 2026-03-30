[CmdletBinding()]
param(
    [ValidateSet('Test', 'Release')]
    [string]$Mode = 'Test',

    [string[]]$Path = @(".\build-release\tools\compiler\Release\arknet.exe"),

    [switch]$Recurse,

    [string]$TestCertSubject = 'CN=ArkNet Dev',
    [string]$TestCertExportDir = '.\.signing',
    [bool]$TrustTestCert = $true,

    [string]$ReleasePfxPath,
    [securestring]$ReleasePfxPassword,
    [string]$TimestampUrl,

    [string]$SignToolPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-PlainText {
    param(
        [Parameter(Mandatory)]
        [securestring]$Value
    )

    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
    try {
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

function Resolve-SignTargets {
    param(
        [Parameter(Mandatory)]
        [string[]]$InputPath,

        [switch]$Recurse
    )

    $files = New-Object System.Collections.Generic.List[string]

    foreach ($entry in $InputPath) {
        $item = Get-Item -LiteralPath $entry

        if ($item.PSIsContainer) {
            $children = Get-ChildItem -LiteralPath $item.FullName -File -Recurse:$Recurse |
                Where-Object { $_.Extension -in '.exe', '.dll' }

            foreach ($child in $children) {
                $files.Add($child.FullName)
            }

            continue
        }

        if ($item.Extension -notin '.exe', '.dll') {
            throw "Unsupported file type for signing: $($item.FullName)"
        }

        $files.Add($item.FullName)
    }

    $resolved = $files | Sort-Object -Unique

    if (-not $resolved -or $resolved.Count -eq 0) {
        throw "No signable files were found."
    }

    $resolved
}

function Find-SignTool {
    param(
        [string]$ExplicitPath
    )

    if ($ExplicitPath) {
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    $command = Get-Command signtool.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($command) {
        return $command.Source
    }

    $roots = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'),
        (Join-Path $env:ProgramFiles 'Windows Kits\10\bin')
    ) | Where-Object { $_ -and (Test-Path $_) }

    foreach ($root in $roots) {
        $match = Get-ChildItem -Path $root -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1

        if ($match) {
            return $match.FullName
        }
    }

    throw "signtool.exe not found. Install the Windows SDK for release signing, or pass -SignToolPath."
}

function Get-OrCreate-TestCertificate {
    param(
        [Parameter(Mandatory)]
        [string]$Subject,

        [Parameter(Mandatory)]
        [string]$ExportDir,

        [bool]$Trust = $true
    )

    $cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert |
        Where-Object { $_.Subject -eq $Subject } |
        Sort-Object NotBefore -Descending |
        Select-Object -First 1

    if (-not $cert) {
        $cert = New-SelfSignedCertificate `
            -Type CodeSigningCert `
            -Subject $Subject `
            -KeyAlgorithm RSA `
            -KeyLength 3072 `
            -HashAlgorithm SHA256 `
            -KeyExportPolicy Exportable `
            -CertStoreLocation 'Cert:\CurrentUser\My' `
            -NotAfter (Get-Date).AddYears(2)
    }

    New-Item -ItemType Directory -Force -Path $ExportDir | Out-Null

    $safeName = ($Subject -replace '[^a-zA-Z0-9\.-]', '_')
    $cerPath = Join-Path $ExportDir "$safeName.cer"

    if (-not (Test-Path -LiteralPath $cerPath)) {
        Export-Certificate -Cert $cert -FilePath $cerPath | Out-Null
    }

    if ($Trust) {
        Import-Certificate -FilePath $cerPath -CertStoreLocation 'Cert:\CurrentUser\Root' | Out-Null
        Import-Certificate -FilePath $cerPath -CertStoreLocation 'Cert:\CurrentUser\TrustedPublisher' | Out-Null
    }

    [pscustomobject]@{
        Certificate = $cert
        CertificatePath = $cerPath
    }
}

function Sign-TestFile {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(Mandatory)]
        [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate
    )

    $result = Set-AuthenticodeSignature `
        -FilePath $FilePath `
        -Certificate $Certificate `
        -HashAlgorithm SHA256 `
        -IncludeChain All

    $verification = Get-AuthenticodeSignature -FilePath $FilePath

    if ($verification.Status -ne 'Valid') {
        throw "Test signing failed for $FilePath. Status: $($verification.Status) - $($verification.StatusMessage)"
    }

    [pscustomobject]@{
        Mode = 'Test'
        File = $FilePath
        Status = $verification.Status
        Subject = $Certificate.Subject
        Thumbprint = $Certificate.Thumbprint
    }
}

function Sign-ReleaseFile {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(Mandatory)]
        [string]$ToolPath,

        [Parameter(Mandatory)]
        [string]$PfxPath,

        [Parameter(Mandatory)]
        [securestring]$PfxPassword,

        [Parameter(Mandatory)]
        [string]$TimestampUrl
    )

    $resolvedPfx = (Resolve-Path -LiteralPath $PfxPath).Path
    $plainPassword = ConvertTo-PlainText -Value $PfxPassword

    try {
        $signArgs = @(
            'sign'
            '/fd', 'SHA256'
            '/td', 'SHA256'
            '/tr', $TimestampUrl
            '/f', $resolvedPfx
            '/p', $plainPassword
            $FilePath
        )

        & $ToolPath @signArgs
        if ($LASTEXITCODE -ne 0) {
            throw "signtool sign failed for $FilePath with exit code $LASTEXITCODE"
        }

        $verifyArgs = @(
            'verify'
            '/pa'
            '/v'
            $FilePath
        )

        & $ToolPath @verifyArgs
        if ($LASTEXITCODE -ne 0) {
            throw "signtool verify failed for $FilePath with exit code $LASTEXITCODE"
        }

        $verification = Get-AuthenticodeSignature -FilePath $FilePath
        if ($verification.Status -ne 'Valid') {
            throw "Release signing verification failed for $FilePath. Status: $($verification.Status) - $($verification.StatusMessage)"
        }

        [pscustomobject]@{
            Mode = 'Release'
            File = $FilePath
            Status = $verification.Status
            Subject = $verification.SignerCertificate.Subject
            Thumbprint = $verification.SignerCertificate.Thumbprint
        }
    }
    finally {
        if ($plainPassword) {
            $plainPassword = $null
        }
    }
}

$targets = Resolve-SignTargets -InputPath $Path -Recurse:$Recurse
$results = New-Object System.Collections.Generic.List[object]

switch ($Mode) {
    'Test' {
        $testSigning = Get-OrCreate-TestCertificate `
            -Subject $TestCertSubject `
            -ExportDir $TestCertExportDir `
            -Trust:$TrustTestCert

        foreach ($target in $targets) {
            $results.Add((Sign-TestFile -FilePath $target -Certificate $testSigning.Certificate))
        }

        Write-Host "Test certificate: $($testSigning.Certificate.Subject)"
        Write-Host "Exported CER: $($testSigning.CertificatePath)"
    }

    'Release' {
        if (-not $ReleasePfxPath) {
            throw "Release mode requires -ReleasePfxPath."
        }

        if (-not $ReleasePfxPassword) {
            throw "Release mode requires -ReleasePfxPassword."
        }

        if (-not $TimestampUrl) {
            throw "Release mode requires -TimestampUrl."
        }

        $tool = Find-SignTool -ExplicitPath $SignToolPath

        foreach ($target in $targets) {
            $results.Add((Sign-ReleaseFile `
                -FilePath $target `
                -ToolPath $tool `
                -PfxPath $ReleasePfxPath `
                -PfxPassword $ReleasePfxPassword `
                -TimestampUrl $TimestampUrl))
        }
    }
}

$results | Format-Table -AutoSize