module attributes {ark.build.unit = "/home/mrkj/ark/benchmarks/bench_modadd.ark"} {
  llvm.func @printNewline()
  llvm.func @printI32(i32)
  llvm.func @printSpace()
  llvm.func @printStr(!llvm.struct<(ptr, i64)>)
  llvm.mlir.global private constant @".str.8315721746935737611"("ark checksum:\00") {addr_space = 0 : i32}
  llvm.func @main() -> i32 attributes {ark.domain = "host", llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %1 = llvm.mlir.constant(13 : i64) : i64
    %2 = llvm.mlir.addressof @".str.8315721746935737611" : !llvm.ptr
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(1000000007 : i32) : i32
    %5 = llvm.mlir.constant(5000000 : i32) : i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i64 : (i64) -> !llvm.ptr
    %8 = llvm.alloca %6 x i32 : (i64) -> !llvm.ptr
    %9 = llvm.alloca %6 x i32 : (i64) -> !llvm.ptr
    %10 = llvm.alloca %6 x i32 : (i64) -> !llvm.ptr
    llvm.store %5, %10 : i32, !llvm.ptr
    llvm.store %4, %9 : i32, !llvm.ptr
    llvm.store %3, %8 : i32, !llvm.ptr
    %11 = llvm.load %10 : !llvm.ptr -> i32
    %12 = llvm.sext %3 : i32 to i64
    %13 = llvm.sext %11 : i32 to i64
    llvm.store %12, %7 : i64, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %14 = llvm.load %7 : !llvm.ptr -> i64
    %15 = llvm.icmp "slt" %14, %13 : i64
    llvm.cond_br %15, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %16 = llvm.load %8 : !llvm.ptr -> i32
    %17 = llvm.load %7 : !llvm.ptr -> i64
    %18 = llvm.sext %16 : i32 to i64
    %19 = llvm.add %18, %17 : i64
    %20 = llvm.load %9 : !llvm.ptr -> i32
    %21 = llvm.sext %20 : i32 to i64
    %22 = llvm.srem %19, %21 : i64
    %23 = llvm.trunc %22 : i64 to i32
    llvm.store %23, %8 : i32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %24 = llvm.load %7 : !llvm.ptr -> i64
    %25 = llvm.add %24, %6 : i64
    llvm.store %25, %7 : i64, !llvm.ptr
    llvm.br ^bb1
  ^bb4:  // pred: ^bb1
    %26 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
    %27 = llvm.insertvalue %26, %0[0] : !llvm.struct<(ptr, i64)> 
    %28 = llvm.insertvalue %1, %27[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%28) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printSpace() : () -> ()
    %29 = llvm.load %8 : !llvm.ptr -> i32
    llvm.call @printI32(%29) : (i32) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return %3 : i32
  }
}
