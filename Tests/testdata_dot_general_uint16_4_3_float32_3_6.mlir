"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui16>, tensor<3x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui16>) -> tensor<4x3xf32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf32>) -> tensor<3x6xf32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    "func.return"(%7) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui16>, tensor<3x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 1, 6], [3, 1, 0], [3, 3, 2], [2, 4, 2]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[0.481567651, 1.21564889, -0.83783406, -6.77650261, -2.61862063, -1.68150115], [-1.98105204, -1.82323277, -2.40821362, -2.40491676, -0.84722644, -2.96528101], [-4.80308437, -3.1644187, 0.0673174784, -4.37394953, 5.07295513, -3.3313961]]> : tensor<3x6xf32>}> : () -> tensor<3x6xf32>
    "func.return"(%1, %2) : (tensor<4x3xui16>, tensor<3x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-29.836422, -18.3784466, -3.67997694, -42.2016182, 24.3532639, -26.3166599], [-0.536349058, 1.8237139, -4.92171574, -22.7344246, -8.7030878, -8.0097847], [-14.1046219, -8.15158939, -9.60350799, -36.2921562, -0.251630783, -20.603138], [-16.5672417, -11.1904707, -11.1738873, -31.9205704, 1.51976299, -21.886919]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%0) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

