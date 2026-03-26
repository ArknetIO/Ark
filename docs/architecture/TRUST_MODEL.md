graph TD
    %% Roles
    Dev((Developer))
    Prov((Provider/Runner))
    Ver((Verifier/Client))

    %% Data Objects
    Source[Source Code + Flags]
    Plan[Canonical Build Plan]
    Artifact[Signed Artifact\n(Binary + Capsule)]
    
    ExecEnv[Execution Environment\n(Hardware + Runtime)]
    Output[Computation Output]
    Receipt[Execution Receipt]

    %% Keys
    DevKey[Dev Signing Key]
    ProvKey[Provider Key]

    %% Flow: The BUILD Phase (Static Truth)
    Dev -->|Writes| Source
    Source -->|Hashed| Plan
    Plan -->|Compiled & Signed| Artifact
    DevKey -.->|Signs| Artifact

    %% Flow: The RUN Phase (Dynamic Truth)
    Artifact -->|Transferred| Prov
    Prov -->|Provisions| ExecEnv
    Artifact -->|Executes on| ExecEnv
    ExecEnv -->|Produces| Output
    ExecEnv -->|Generates| Receipt
    
    ProvKey -.->|Signs| Receipt
    
    %% The Link
    Artifact -.->|PlanHash| Receipt

    %% Flow: The VERIFY Phase (Settlement)
    Receipt -->|Submitted to| Ver
    Artifact -->|Reference| Ver
    Output -->|Hash Check| Ver
    
    subgraph Verification Logic
        Check1{Sig Valid?}
        Check2{PlanHash Match?}
        Check3{SLO Met?}
        Check4{Output Hash Match?}
    end

    Ver --> Check1
    Check1 -->|Yes| Check2
    Check2 -->|Yes| Check3
    Check3 -->|Yes| Check4
    Check4 -->|Yes| Trusted[Settled / Trusted]
    Check4 -->|No| Slashed[Slashed / Rejected]