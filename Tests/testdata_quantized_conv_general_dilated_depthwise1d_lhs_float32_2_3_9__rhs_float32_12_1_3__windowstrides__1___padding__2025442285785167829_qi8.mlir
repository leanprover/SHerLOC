"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-1.96511757, -5.55958605, -1.50374639, 1.2120446, 3.21186447, 1.02349842, 5.26346111, -0.576352775, -2.11138129], [2.14414096, -1.50321436, 2.67583466, 4.80866337, 1.8576318, -4.17470455, -0.173302904, -6.73420906, -0.265416831], [-0.179974869, 2.97624516, -4.66533756, 2.16204596, 3.08222961, 2.36635828, -0.860805928, 3.5810957, 3.5065918]], [[1.85445964, 3.97995543, 0.977302432, 2.28219128, 1.52758765, 3.04823041, 0.792750239, -2.62015224, 0.797125578], [1.68729472, 4.16192293, -0.865549385, -1.65564895, 1.02495849, 3.03293586, 1.41485655, 2.36943197, -2.79530549], [2.75504136, -0.195227623, -3.81070256, 0.56287992, -3.70327806, -1.78669858, 0.2387449, -3.83501959, -2.27439022]]]> : tensor<2x3x9xf32>}> : () -> tensor<2x3x9xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[-1.80246711, 1.96004879, -0.605032503]], [[4.45640278, -5.46205235, 1.1135869]], [[-4.13330603, -3.73956728, -4.24840975]], [[3.73067904, -4.542027, -2.5989809]], [[-0.119584844, -0.0161871985, -0.134076193]], [[1.08771026, -2.79720259, 2.24169183]], [[-2.83143401, 0.309991777, 2.02455497]], [[-1.20667934, -0.505252481, 2.85701632]], [[0.257799268, 1.51054871, -1.96405339]], [[-0.451475561, 0.847560584, -1.11103594]], [[-0.222673655, 4.257411, -3.94667459]], [[-0.5362221, 2.006420e+00, 0.046573516]]]> : tensor<12x1x3xf32>}> : () -> tensor<12x1x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<"0x00000000000000009A60803F9A60803F9A60803F9A60803F00000000000000009A60803F9A60803F6F96FF3F6F96FF3F9A60803F9A60803F000000000000000000000000000000000000000000000000000000000000000000000000000000009A60803F9A60803F9A60803F9A60803F000000000000000000000000000000000000000000000000000000006F96FF3F9A60803F6F96FF3F9A60803F9A60803F00000000000000009A60803FC00EA83FC00EA83F99B89E3E0000000000000000000000009A60803F9A60803F9A60803F000000000000000000000000000000009A60803F24B6823E9A60803F230EA13F230EA13F24B6823E9A60803F0E13593F000000000E13593F0E13593F0E13593F000000000E13593F9A60803F000000009A60803F9A60803F9A60803F000000009A60803F9A60803F0EBB3A3D7236863F7236863F9A60803F0EBB3A3D7236863F9A60803F97C0793F9A60803F9A60803F9A60803FD3114B3F00000000E540FD3F6F96FF3FE540FD3F6F96FF3FBFBEE43F9A60803FD311CB3F000000000000000000000000000000000000000000000000000000009A60803F9A60803F97C0793F9A60803F9A60803F9A60803FD3114B3F000000000000000000000000000000000000000000000000000000009A60803F9A60803F9A60803F9A60803F6F96FF3F6F96FF3F9A60803F99B89E3E000000009A60803FC00EA83FC00EA83FC00EA83F99B89E3E00000000000000009A60803F9A60803F9A60803F9A60803F0000000024B6823E000000005EB7103F7262153E00000000F9BF723E24B6823D0000000000000000F9BFF23E00000000000000005D674D3E0000000000000000000000005EB7103F0000000000000000F9BF723E0000000000000000AB13E03C5EB7103F000000007262153CF9BF723E00000000"> : tensor<2x12x7xf32>}> : () -> tensor<2x12x7xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<12x1x3xf32>) -> tensor<12x1x3x!quant.uniform<i8:f32, 0.0039196598763559382:-128>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<2x3x9xf32>) -> tensor<2x3x9x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %5 = "stablehlo.convolution"(%4, %3) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, f, 0]x[o, i, 0]->[b, f, 0]>, feature_group_count = 3 : i64}> : (tensor<2x3x9x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>, tensor<12x1x3x!quant.uniform<i8:f32, 0.0039196598763559382:-128>>) -> tensor<2x12x7x!quant.uniform<i32:f32, 1.5371135492359336E-5>>
    %6 = "stablehlo.uniform_quantize"(%5) : (tensor<2x12x7x!quant.uniform<i32:f32, 1.5371135492359336E-5>>) -> tensor<2x12x7x!quant.uniform<i8:f32, 0.0091177098891314333:-128>>
    %7 = "stablehlo.uniform_dequantize"(%6) : (tensor<2x12x7x!quant.uniform<i8:f32, 0.0091177098891314333:-128>>) -> tensor<2x12x7xf32>
    %8 = "stablehlo.custom_call"(%2, %7) <{call_target_name = "check.eq"}> : (tensor<2x12x7xf32>, tensor<2x12x7xf32>) -> tensor<i1>
    "func.return"(%8) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

