"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui16>, tensor<3x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui16>) -> tensor<4x3xf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf16>) -> tensor<3x6xf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    "func.return"(%7) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui16>, tensor<3x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 0, 0], [5, 1, 3], [3, 3, 5], [1, 0, 6]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[3.820310e+00, 2.255860e+00, 5.359380e+00, 6.479490e-01, -2.013670e+00, -2.236330e+00], [-9.077140e-01, -4.492190e-01, 2.250000e+00, 4.054690e+00, 3.130860e+00, 6.831050e-01], [-3.121090e+00, 3.070310e+00, -2.919920e+00, 1.774410e+00, -3.007810e-01, -1.563480e+00]]> : tensor<3x6xf16>}> : () -> tensor<3x6xf16>
    "func.return"(%1, %2) : (tensor<4x3xui16>, tensor<3x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[7.640630e+00, 4.511720e+00, 1.071880e+01, 1.295900e+00, -4.027340e+00, -4.472660e+00], [8.828120e+00, 2.004690e+01, 2.028130e+01, 1.261720e+01, -7.839840e+00, -1.518750e+01], [-6.867180e+00, 2.076560e+01, 8.226560e+00, 2.298440e+01, 1.847660e+00, -1.247660e+01], [-1.490630e+01, 2.067190e+01, -1.215630e+01, 1.129690e+01, -3.818360e+00, -1.161720e+01]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%0) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

