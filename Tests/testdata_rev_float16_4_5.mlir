"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x5xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x5xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4x5xf16>
    %4 = "stablehlo.reverse"(%2) <{dimensions = array<i64: 0>}> : (tensor<4x5xf16>) -> tensor<4x5xf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x5xf16>, tensor<4x5xf16>) -> ()
    "func.return"(%4) : (tensor<4x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-6.968750e+00, -1.627930e+00, -1.103520e+00, 3.359380e+00, 2.449950e-01], [-5.144530e+00, 9.155270e-01, -1.909180e+00, -1.003910e+00, -3.773440e+00], [-8.979490e-01, 8.730460e-01, 3.382810e+00, 2.332030e+00, -1.281250e+00], [-4.339840e+00, 1.500980e+00, 5.791020e-01, 4.203130e+00, 2.228520e+00]]> : tensor<4x5xf16>}> : () -> tensor<4x5xf16>
    "func.return"(%1) : (tensor<4x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.339840e+00, 1.500980e+00, 5.791020e-01, 4.203130e+00, 2.228520e+00], [-8.979490e-01, 8.730460e-01, 3.382810e+00, 2.332030e+00, -1.281250e+00], [-5.144530e+00, 9.155270e-01, -1.909180e+00, -1.003910e+00, -3.773440e+00], [-6.968750e+00, -1.627930e+00, -1.103520e+00, 3.359380e+00, 2.449950e-01]]> : tensor<4x5xf16>}> : () -> tensor<4x5xf16>
    "func.return"(%0) : (tensor<4x5xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

