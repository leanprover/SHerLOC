"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0], [2], [1]]> : tensor<3x1xi32>}> : () -> tensor<3x1xi32>
    %1 = "stablehlo.constant"() <{value = dense<[[3.27917194, 4.79832602, -0.10540507, 3.0128758, -0.239777446], [2.15583777, 1.20524621, -5.21279478, 1.90036821, -5.13092661], [3.04004383, -0.748676181, 2.70175195, 1.52100611, -3.00484538], [3.73827457, 0.19627282, 1.94314909, 1.35509837, -3.43813014], [-3.82101965, -1.52074528, 3.47338939, -1.92120409, 0.261425197], [3.99755049, 1.19325948, -4.27102518, 0.404079616, 2.05883861], [-0.426693857, 3.07045269, 3.41785836, -3.19426727, -4.46876669], [2.52909493, 5.09040689, -0.238279715, -0.449262351, 2.13594961], [-3.88575697, -3.08637547, 3.126508, 1.62036669, -3.84586406], [-1.86297441, -6.78266811, -0.103852905, 2.65204668, 5.41461182]]> : tensor<10x5xf32>}> : () -> tensor<10x5xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0.998835206, 0.998835206, 0.000000e+00], [0.998835206, 0.000000e+00, 0.998835206], [0.998835206, 0.998835206, 0.000000e+00]]> : tensor<3x3xf32>}> : () -> tensor<3x3xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<10x5xf32>) -> tensor<10x5x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %4 = "stablehlo.gather"(%3, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3>}> : (tensor<10x5x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>, tensor<3x1xi32>) -> tensor<3x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %5 = "stablehlo.uniform_quantize"(%4) : (tensor<3x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<3x3x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<3x3x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>) -> tensor<3x3xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<i1>
    "func.return"(%7) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

