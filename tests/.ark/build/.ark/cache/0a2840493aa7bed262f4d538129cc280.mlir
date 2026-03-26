module attributes {ark.build.unit = "/home/mrkj/ark/tests/monolith.ark"} {
  llvm.mlir.global private constant @".str.9592373465895250038"("errors.\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @".str.15632824356166372197"("FAILURE: Found\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @".str.4037845530463115344"("elements.\00") {addr_space = 0 : i32}
  llvm.func @printI32(i32)
  llvm.mlir.global private constant @".str.13476515021349330271"("SUCCESS: GPU Kernel addition verified correctly over\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @".str.4570745336264320552"("... stopping verification (too many errors).\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @".str.17816461888716901353"("got\00") {addr_space = 0 : i32}
  llvm.func @printF32(f32)
  llvm.mlir.global private constant @".str.501897112562995416"(": expected\00") {addr_space = 0 : i32}
  llvm.func @printI64(i64)
  llvm.func @printSpace()
  llvm.mlir.global private constant @".str.10916321145990628239"("Error at index\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @".str.11601353813030804829"("Verifying results...\00") {addr_space = 0 : i32}
  llvm.func @__ark_await(i64) -> i32
  llvm.mlir.global private constant @".str.2479455475611130042"("Launching GPU kernel...\00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @".str.15147977084159862343"("Initializing input tensors...\00") {addr_space = 0 : i32}
  llvm.func @printNewline()
  llvm.func @printStr(!llvm.struct<(ptr, i64)>)
  llvm.mlir.global private constant @".str.6297998075515662775"("=== TEST: GPU KERNEL LAUNCH ===\00") {addr_space = 0 : i32}
  llvm.func @__ark_gpu_alloc_managed(i64) -> !llvm.ptr
  llvm.func @__ark_sys_process_init(i32, !llvm.ptr, !llvm.ptr)
  llvm.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 attributes {ark.domain = "host", llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(7 : i64) : i64
    %1 = llvm.mlir.addressof @".str.9592373465895250038" : !llvm.ptr
    %2 = llvm.mlir.addressof @".str.15632824356166372197" : !llvm.ptr
    %3 = llvm.mlir.constant(9 : i64) : i64
    %4 = llvm.mlir.addressof @".str.4037845530463115344" : !llvm.ptr
    %5 = llvm.mlir.constant(52 : i64) : i64
    %6 = llvm.mlir.addressof @".str.13476515021349330271" : !llvm.ptr
    %7 = llvm.mlir.constant(44 : i64) : i64
    %8 = llvm.mlir.addressof @".str.4570745336264320552" : !llvm.ptr
    %9 = llvm.mlir.constant(5 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(3 : i64) : i64
    %12 = llvm.mlir.addressof @".str.17816461888716901353" : !llvm.ptr
    %13 = llvm.mlir.constant(10 : i64) : i64
    %14 = llvm.mlir.addressof @".str.501897112562995416" : !llvm.ptr
    %15 = llvm.mlir.constant(14 : i64) : i64
    %16 = llvm.mlir.addressof @".str.10916321145990628239" : !llvm.ptr
    %17 = llvm.mlir.constant(3.000000e+01 : f32) : f32
    %18 = llvm.mlir.constant(20 : i64) : i64
    %19 = llvm.mlir.addressof @".str.11601353813030804829" : !llvm.ptr
    %20 = llvm.mlir.constant(23 : i64) : i64
    %21 = llvm.mlir.addressof @".str.2479455475611130042" : !llvm.ptr
    %22 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %23 = llvm.mlir.constant(2.000000e+01 : f32) : f32
    %24 = llvm.mlir.constant(1.000000e+01 : f32) : f32
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(29 : i64) : i64
    %27 = llvm.mlir.addressof @".str.15147977084159862343" : !llvm.ptr
    %28 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %29 = llvm.mlir.constant(31 : i64) : i64
    %30 = llvm.mlir.constant(0 : i64) : i64
    %31 = llvm.mlir.addressof @".str.6297998075515662775" : !llvm.ptr
    %32 = llvm.mlir.constant(4 : i64) : i64
    %33 = llvm.mlir.constant(1024 : i32) : i32
    %34 = llvm.mlir.constant(1 : i64) : i64
    %35 = llvm.alloca %34 x f32 : (i64) -> !llvm.ptr
    %36 = llvm.alloca %34 x f32 : (i64) -> !llvm.ptr
    %37 = llvm.alloca %34 x i64 : (i64) -> !llvm.ptr
    %38 = llvm.alloca %34 x i32 : (i64) -> !llvm.ptr
    %39 = llvm.alloca %34 x i64 : (i64) -> !llvm.ptr
    %40 = llvm.alloca %34 x i64 : (i64) -> !llvm.ptr
    %41 = llvm.alloca %34 x !llvm.ptr : (i64) -> !llvm.ptr
    %42 = llvm.alloca %34 x !llvm.ptr : (i64) -> !llvm.ptr
    %43 = llvm.alloca %34 x !llvm.ptr : (i64) -> !llvm.ptr
    %44 = llvm.alloca %34 x i32 : (i64) -> !llvm.ptr
    llvm.call @__ark_sys_process_init(%arg0, %arg1, %arg2) : (i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.store %33, %44 : i32, !llvm.ptr
    %45 = llvm.load %44 : !llvm.ptr -> i32
    %46 = llvm.sext %45 : i32 to i64
    %47 = llvm.mul %46, %32 : i64
    %48 = llvm.call @__ark_gpu_alloc_managed(%47) : (i64) -> !llvm.ptr
    llvm.store %48, %43 : !llvm.ptr, !llvm.ptr
    %49 = llvm.load %44 : !llvm.ptr -> i32
    %50 = llvm.sext %49 : i32 to i64
    %51 = llvm.mul %50, %32 : i64
    %52 = llvm.call @__ark_gpu_alloc_managed(%51) : (i64) -> !llvm.ptr
    llvm.store %52, %42 : !llvm.ptr, !llvm.ptr
    %53 = llvm.load %44 : !llvm.ptr -> i32
    %54 = llvm.sext %53 : i32 to i64
    %55 = llvm.mul %54, %32 : i64
    %56 = llvm.call @__ark_gpu_alloc_managed(%55) : (i64) -> !llvm.ptr
    llvm.store %56, %41 : !llvm.ptr, !llvm.ptr
    %57 = llvm.getelementptr %31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
    %58 = llvm.insertvalue %57, %28[0] : !llvm.struct<(ptr, i64)> 
    %59 = llvm.insertvalue %29, %58[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%59) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printNewline() : () -> ()
    %60 = llvm.getelementptr %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<30 x i8>
    %61 = llvm.insertvalue %60, %28[0] : !llvm.struct<(ptr, i64)> 
    %62 = llvm.insertvalue %26, %61[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%62) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printNewline() : () -> ()
    %63 = llvm.load %44 : !llvm.ptr -> i32
    %64 = llvm.sext %25 : i32 to i64
    %65 = llvm.sext %63 : i32 to i64
    llvm.store %64, %40 : i64, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %66 = llvm.load %40 : !llvm.ptr -> i64
    %67 = llvm.icmp "slt" %66, %65 : i64
    llvm.cond_br %67, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %68 = llvm.load %40 : !llvm.ptr -> i64
    %69 = llvm.load %43 : !llvm.ptr -> !llvm.ptr
    %70 = llvm.getelementptr %69[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %24, %70 : f32, !llvm.ptr
    %71 = llvm.load %40 : !llvm.ptr -> i64
    %72 = llvm.load %42 : !llvm.ptr -> !llvm.ptr
    %73 = llvm.getelementptr %72[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %23, %73 : f32, !llvm.ptr
    %74 = llvm.load %40 : !llvm.ptr -> i64
    %75 = llvm.load %41 : !llvm.ptr -> !llvm.ptr
    %76 = llvm.getelementptr %75[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %22, %76 : f32, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %77 = llvm.load %40 : !llvm.ptr -> i64
    %78 = llvm.add %77, %34 : i64
    llvm.store %78, %40 : i64, !llvm.ptr
    llvm.br ^bb1
  ^bb4:  // pred: ^bb1
    %79 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
    %80 = llvm.insertvalue %79, %28[0] : !llvm.struct<(ptr, i64)> 
    %81 = llvm.insertvalue %20, %80[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%81) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printNewline() : () -> ()
    %82 = llvm.load %43 : !llvm.ptr -> !llvm.ptr
    %83 = llvm.load %42 : !llvm.ptr -> !llvm.ptr
    %84 = llvm.load %41 : !llvm.ptr -> !llvm.ptr
    llvm.call @add_kernel(%82, %83, %84) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.store %30, %39 : i64, !llvm.ptr
    %85 = llvm.load %39 : !llvm.ptr -> i64
    %86 = llvm.call @__ark_await(%85) : (i64) -> i32
    %87 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
    %88 = llvm.insertvalue %87, %28[0] : !llvm.struct<(ptr, i64)> 
    %89 = llvm.insertvalue %18, %88[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%89) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.store %25, %38 : i32, !llvm.ptr
    %90 = llvm.load %44 : !llvm.ptr -> i32
    %91 = llvm.sext %90 : i32 to i64
    llvm.store %64, %37 : i64, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb7
    %92 = llvm.load %37 : !llvm.ptr -> i64
    %93 = llvm.icmp "slt" %92, %91 : i64
    llvm.cond_br %93, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %94 = llvm.load %41 : !llvm.ptr -> !llvm.ptr
    %95 = llvm.load %37 : !llvm.ptr -> i64
    %96 = llvm.getelementptr %94[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.load %96 : !llvm.ptr -> f32
    llvm.store %97, %36 : f32, !llvm.ptr
    llvm.store %17, %35 : f32, !llvm.ptr
    %98 = llvm.load %36 : !llvm.ptr -> f32
    %99 = llvm.load %35 : !llvm.ptr -> f32
    %100 = llvm.fcmp "une" %98, %99 : f32
    llvm.cond_br %100, ^bb9, ^bb10
  ^bb7:  // pred: ^bb11
    %101 = llvm.load %37 : !llvm.ptr -> i64
    %102 = llvm.add %101, %34 : i64
    llvm.store %102, %37 : i64, !llvm.ptr
    llvm.br ^bb5
  ^bb8:  // pred: ^bb5
    %103 = llvm.load %38 : !llvm.ptr -> i32
    %104 = llvm.icmp "eq" %103, %25 : i32
    llvm.cond_br %104, ^bb15, ^bb16
  ^bb9:  // pred: ^bb6
    %105 = llvm.getelementptr %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %106 = llvm.insertvalue %105, %28[0] : !llvm.struct<(ptr, i64)> 
    %107 = llvm.insertvalue %15, %106[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%107) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printSpace() : () -> ()
    %108 = llvm.load %37 : !llvm.ptr -> i64
    llvm.call @printI64(%108) : (i64) -> ()
    llvm.call @printSpace() : () -> ()
    %109 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<11 x i8>
    %110 = llvm.insertvalue %109, %28[0] : !llvm.struct<(ptr, i64)> 
    %111 = llvm.insertvalue %13, %110[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%111) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printSpace() : () -> ()
    %112 = llvm.load %35 : !llvm.ptr -> f32
    llvm.call @printF32(%112) : (f32) -> ()
    llvm.call @printSpace() : () -> ()
    %113 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %114 = llvm.insertvalue %113, %28[0] : !llvm.struct<(ptr, i64)> 
    %115 = llvm.insertvalue %11, %114[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%115) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printSpace() : () -> ()
    %116 = llvm.load %36 : !llvm.ptr -> f32
    llvm.call @printF32(%116) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    %117 = llvm.load %38 : !llvm.ptr -> i32
    %118 = llvm.add %117, %10 : i32
    llvm.store %118, %38 : i32, !llvm.ptr
    %119 = llvm.load %38 : !llvm.ptr -> i32
    %120 = llvm.icmp "sgt" %119, %9 : i32
    llvm.cond_br %120, ^bb12, ^bb13
  ^bb10:  // pred: ^bb6
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb10, ^bb14
    llvm.br ^bb7
  ^bb12:  // pred: ^bb9
    %121 = llvm.getelementptr %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<45 x i8>
    %122 = llvm.insertvalue %121, %28[0] : !llvm.struct<(ptr, i64)> 
    %123 = llvm.insertvalue %7, %122[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%123) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return %10 : i32
  ^bb13:  // pred: ^bb9
    llvm.br ^bb14
  ^bb14:  // pred: ^bb13
    llvm.br ^bb11
  ^bb15:  // pred: ^bb8
    %124 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<53 x i8>
    %125 = llvm.insertvalue %124, %28[0] : !llvm.struct<(ptr, i64)> 
    %126 = llvm.insertvalue %5, %125[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%126) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printSpace() : () -> ()
    %127 = llvm.load %44 : !llvm.ptr -> i32
    llvm.call @printI32(%127) : (i32) -> ()
    llvm.call @printSpace() : () -> ()
    %128 = llvm.getelementptr %4[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x i8>
    %129 = llvm.insertvalue %128, %28[0] : !llvm.struct<(ptr, i64)> 
    %130 = llvm.insertvalue %3, %129[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%130) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return %25 : i32
  ^bb16:  // pred: ^bb8
    %131 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
    %132 = llvm.insertvalue %131, %28[0] : !llvm.struct<(ptr, i64)> 
    %133 = llvm.insertvalue %15, %132[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%133) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printSpace() : () -> ()
    %134 = llvm.load %38 : !llvm.ptr -> i32
    llvm.call @printI32(%134) : (i32) -> ()
    llvm.call @printSpace() : () -> ()
    %135 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i8>
    %136 = llvm.insertvalue %135, %28[0] : !llvm.struct<(ptr, i64)> 
    %137 = llvm.insertvalue %0, %136[1] : !llvm.struct<(ptr, i64)> 
    llvm.call @printStr(%137) : (!llvm.struct<(ptr, i64)>) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return %10 : i32
  }
  llvm.func @__ark_gpu_launch(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr)
  llvm.mlir.global private constant @".str.3599770855740484229"("add_kernel\00") {addr_space = 0 : i32}
  llvm.func @add_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(3 : i32) : i32
    %2 = llvm.mlir.constant(32 : i32) : i32
    %3 = llvm.mlir.addressof @".str.3599770855740484229" : !llvm.ptr
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.array<3 x ptr> : (i32) -> !llvm.ptr
    %6 = llvm.alloca %4 x !llvm.ptr : (i32) -> !llvm.ptr
    llvm.store %arg0, %6 : !llvm.ptr, !llvm.ptr
    %7 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
    llvm.store %6, %7 : !llvm.ptr, !llvm.ptr
    %8 = llvm.alloca %4 x !llvm.ptr : (i32) -> !llvm.ptr
    llvm.store %arg1, %8 : !llvm.ptr, !llvm.ptr
    %9 = llvm.getelementptr %5[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
    llvm.store %8, %9 : !llvm.ptr, !llvm.ptr
    %10 = llvm.alloca %4 x !llvm.ptr : (i32) -> !llvm.ptr
    llvm.store %arg2, %10 : !llvm.ptr, !llvm.ptr
    %11 = llvm.getelementptr %5[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
    llvm.store %10, %11 : !llvm.ptr, !llvm.ptr
    %12 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<11 x i8>
    llvm.call @__ark_gpu_launch(%12, %5, %1, %2, %4, %4, %2, %4, %4, %0) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> ()
    llvm.return
  }
  gpu.module @ark.gpu.module {
    gpu.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) kernel attributes {ark.gpu.kernel} {
      %0 = llvm.mlir.constant(1024 : i64) : i64
      %1 = llvm.mlir.constant(1 : i64) : i64
      %2 = llvm.alloca %1 x i64 : (i64) -> !llvm.ptr
      %3 = llvm.alloca %1 x !llvm.ptr<1> : (i64) -> !llvm.ptr
      llvm.store %arg0, %3 : !llvm.ptr<1>, !llvm.ptr
      %4 = llvm.alloca %1 x !llvm.ptr<1> : (i64) -> !llvm.ptr
      llvm.store %arg1, %4 : !llvm.ptr<1>, !llvm.ptr
      %5 = llvm.alloca %1 x !llvm.ptr<1> : (i64) -> !llvm.ptr
      llvm.store %arg2, %5 : !llvm.ptr<1>, !llvm.ptr
      %thread_id_x = gpu.thread_id  x
      %block_id_x = gpu.block_id  x
      %block_dim_x = gpu.block_dim  x
      %grid_dim_x = gpu.grid_dim  x
      %6 = arith.muli %block_id_x, %block_dim_x : index
      %7 = arith.addi %6, %thread_id_x : index
      %8 = arith.index_cast %7 : index to i64
      %9 = arith.muli %grid_dim_x, %block_dim_x : index
      %10 = arith.index_cast %9 : index to i64
      cf.br ^bb1(%8 : i64)
    ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb2
      %12 = arith.cmpi slt, %11, %0 : i64
      cf.cond_br %12, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      llvm.store %11, %2 : i64, !llvm.ptr
      %13 = llvm.load %3 : !llvm.ptr -> !llvm.ptr<1>
      %14 = llvm.load %2 : !llvm.ptr -> i64
      %15 = llvm.getelementptr %13[%14] : (!llvm.ptr<1>, i64) -> !llvm.ptr, f32
      %16 = llvm.load %15 : !llvm.ptr -> f32
      %17 = llvm.load %4 : !llvm.ptr -> !llvm.ptr<1>
      %18 = llvm.getelementptr %17[%14] : (!llvm.ptr<1>, i64) -> !llvm.ptr, f32
      %19 = llvm.load %18 : !llvm.ptr -> f32
      %20 = llvm.fadd %16, %19 : f32
      %21 = llvm.load %5 : !llvm.ptr -> !llvm.ptr<1>
      %22 = llvm.getelementptr %21[%14] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %20, %22 : f32, !llvm.ptr<1>
      %23 = arith.addi %11, %10 : i64
      cf.br ^bb1(%23 : i64)
    ^bb3:  // pred: ^bb1
      gpu.return
    }
  }
}
