"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui8>, tensor<3x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui8>) -> tensor<4x3xf64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf64>) -> tensor<3x6xf64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    "func.return"(%7) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui8>, tensor<3x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 0, 5], [0, 3, 5], [5, 2, 0], [2, 1, 1]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[-4.3886811303306077, -7.1534168587462892, -6.6619289284585239, 2.1838990687896942, -2.7471214742479537, -1.2088552070550003], [-1.326909111780642, -3.2791645718163505, -2.3148246916008381, -0.050698021974274678, -1.011258651905828, 1.4946544363617982], [1.3584636024988166, -0.43609970000415194, -1.6081534488018772, 0.25021662296343494, -1.283272445359728, -2.5618008962455652]]> : tensor<3x6xf64>}> : () -> tensor<3x6xf64>
    "func.return"(%1, %2) : (tensor<4x3xui8>, tensor<3x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[2.4036368821634753, -9.3339153587670491, -14.70269617246791, 3.434982183606869, -9.1634837010465943, -14.017859688282826], [2.8115906771521573, -12.01799221546981, -14.9852413188119, 1.0989890488943508, -9.4501381825161239, -8.3250411721424307], [-24.597223875214322, -42.325413437364148, -37.939294025494291, 10.818099299999922, -15.758124675051425, -3.0549671625514052], [-8.7458077699430419, -18.022097989313082, -17.246835997319764, 4.5673167385685485, -7.7887740457614631, -3.4848568739937678]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%0) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

