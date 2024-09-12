"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xf16>, tensor<3x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf32>) -> tensor<3x6xf32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    "func.return"(%7) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xf16>, tensor<3x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[6.289060e-01, -1.862300e+00, 4.878910e+00], [-1.801760e+00, -4.636230e-01, 2.792970e-01], [1.379880e+00, 5.246090e+00, -9.033200e-01], [-7.660150e+00, -2.015630e+00, -7.167960e-01]]> : tensor<4x3xf16>}> : () -> tensor<4x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[1.03904819, -0.390662134, 0.586938202, 0.189655736, -3.34397602, -1.60065711], [-1.24223864, -4.05648661, 4.49510813, 0.90029788, -1.29326081, 3.30721259], [-1.39605582, 1.74161911, 1.90680158, -0.809361219, 1.66872275, 0.672101855]]> : tensor<3x6xf32>}> : () -> tensor<3x6xf32>
    "func.return"(%1, %2) : (tensor<4x3xf16>, tensor<3x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3.8443346, 15.8059206, 1.30097473, -5.50615072, 8.446940e+00, -3.88657904], [-1.68609679, 3.07098794, -2.60899258, -0.985164582, 7.09068965, 1.53841257], [-3.822050e+00, -23.3930168, 22.6692123, 5.71586227, -12.9062538, 14.5341043], [-4.4546957, 9.9205017, -14.9232807, -2.68730783, 27.0259743, 5.11342287]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%0) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

