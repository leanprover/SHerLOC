"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf32>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xf32>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf32>) -> tensor<3x6xf32>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    "func.return"(%7) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3, -2, -1], [0, -3, 1], [-6, 0, 0], [2, -6, -4]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[-3.99020648, 2.58919454, 0.433581144, 3.68860888, 2.73794389, -2.00574136], [-2.59321427, -1.43049741, -3.01954341, 0.79414606, 0.508764327, -0.239404649], [-0.140727609, -4.93321419, -0.293230742, -4.03748178, -1.0149287, 1.85012126]]> : tensor<3x6xf32>}> : () -> tensor<3x6xf32>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[17.2977753, 0.0266251564, 5.03157377, -8.61663627, -8.21643161, 4.64591217], [7.63891554, -0.641721725, 8.76539897, -6.419920e+00, -2.54122162, 2.56833506], [23.9412384, -15.5351677, -2.60148692, -22.1316528, -16.4276638, 12.0344486], [8.14178276, 33.4942322, 20.1573448, 18.7622681, 6.48301649, -9.975540e+00]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%0) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

