"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xf32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf32>) -> tensor<3x6xf32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    "func.return"(%7) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[5, 2, -2], [1, 0, -1], [-5, 2, -2], [-4, -1, -5]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.94647026, 0.734722912, -2.58595443, 1.07965136, 1.0958364, 2.70218849], [-3.40507245, -1.9028939, 7.62151289, 3.63030744, -0.233068526, -1.80128574], [2.33670807, 1.56321156, -0.354040384, 1.42250979, -4.28100061, 1.82065916]]> : tensor<3x6xf32>}> : () -> tensor<3x6xf32>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-26.2159119, -3.25859642, 3.02133417, 9.81385231, 13.5750465, 6.26705265], [-5.28317833, -0.828488647, -2.23191404, -0.342858434, 5.37683678, 0.881529331], [3.24879026, -10.6058254, 28.8808784, -0.982661485, 2.61668205, -20.7548332], [3.50741291, -8.85205554, 4.49250698, -15.0614614, 17.2547264, -18.1107635]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%0) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

