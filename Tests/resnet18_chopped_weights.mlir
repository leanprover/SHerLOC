"builtin.module"() <{sym_name = "jit__unnamed_wrapped_function_"}> ({
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<1x3x224x224xf32>) -> (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg5: tensor<1x3x224x224xf32>):
    %16 = "stablehlo.constant"() <{value = dense<[0.00276811025, -0.0257689245, 2.12544663E-7, -0.0846051499, 2.11205915E-8, 4.96906461E-4, -0.0224083271, -1.15818956E-7, -4.823850e-03, 2.75073347E-7, 3.958230e-02, 0.0319936387, -0.0374896601, -1.37163477E-6, 0.00660019321, 4.378190e-03, 0.0647971481, 0.11175999, 0.0360015705, -0.0750752166, -0.03824009, 0.0843578502, -5.228700e-02, -0.0117988894, 0.00130188058, 0.0321722962, -0.0177843049, -0.0910085887, 0.113187239, -0.0416320041, 0.00873024761, 0.0296931695, -0.0705021694, -0.00348469778, 0.109771468, -0.00173411821, -5.94229412E-8, 0.0293303896, -7.85527287E-9, 0.00673204474, -0.00370999542, 0.0160279572, -0.0278826058, 0.0265925378, 0.0284745526, -0.12734659, 0.0446168408, 0.0263288375, 2.14538591E-8, -1.704500e-02, -0.00356168114, -0.0458412617, 0.0638761446, 0.0152198272, -0.0385114551, -0.0164278317, -0.016568929, 0.0560574941, -0.0803062319, -0.00266457512, -0.0417176671, 0.126112729, -0.0492369682, -0.0132609205]> : tensor<64xf32>}> : () -> tensor<64xf32>
    %17 = "stablehlo.constant"() <{value = dense<[1.01694489, 3.71674347, 5.81334356E-11, 3.28254271, 1.71074404E-13, 0.658226967, 4.37006235, 6.60045282E-12, 0.915522992, 1.93175254E-9, 4.12558556, 2.74399233, 2.8390913, 4.79658588E-8, 11.0722713, 0.500745952, 2.23128176, 4.82570696, 2.69861364, 9.36995506, 3.73391747, 5.48429585, 5.7126689, 0.445444882, 0.436275303, 7.15633583, 13.7179089, 5.25117493, 6.81737518, 1.67235756, 1.65343034, 1.23245978, 4.90762854, 3.07305121, 4.23838568, 4.99363518, 1.44646307E-12, 1.52116203, 1.03519833E-13, 0.351344079, 0.17024748, 1.42054474, 1.90848303, 2.15124035, 2.66084933, 4.84443378, 1.92971194, 1.49994361, 2.94806145E-13, 1.53064024, 0.365027189, 2.93755412, 5.46641159, 0.707924544, 3.33150721, 0.771802961, 2.40678358, 6.5213666, 4.12625027, 1.05063522, 2.95303202, 11.3656216, 4.76904678, 1.65587807]> : tensor<64xf32>}> : () -> tensor<64xf32>
    %18 = "stablehlo.constant"() <{value = dense<[0.234872743, 0.266257942, -5.10959595E-8, 0.518699706, 3.44040196E-9, 0.222385287, 0.422887057, 1.31532403E-7, 0.25093165, 1.5152026E-6, 0.316871643, 0.250491828, 0.378926098, 1.08618351E-5, 2.752640e-01, 0.236741036, 0.242021769, 0.395314813, 0.469346285, 0.2908957, 0.272684187, 0.27802828, 0.290692091, 0.206927493, 0.258990377, 0.278710574, 0.291149527, 0.316013753, 0.388891488, 0.304111898, 0.267757207, 0.210925162, 0.287084132, 0.332426429, 0.42672804, 0.373260558, 7.48037578E-8, 0.19067812, 1.47401256E-8, 0.223029822, 0.179079413, 0.248600766, 0.27399528, 0.259228647, 0.294202209, 0.299236417, 0.223688841, 0.262799472, 2.20011476E-8, 0.266098082, 0.220890298, 0.284285516, 0.330723315, 0.226809531, 0.365380913, 0.21229881, 0.239653021, 0.24949576, 0.525830686, 0.248247579, 0.295652747, 0.258776665, 0.4832564, 0.26670444]> : tensor<64xf32>}> : () -> tensor<64xf32>
    %19 = "stablehlo.constant"() <{value = dense<[0.230717152, 0.253822476, -1.05429808E-6, -0.664388895, -1.65705547E-8, 0.161521927, 0.454503953, -4.301950e-07, 0.300513744, -8.005240e-06, 0.349418074, 0.311480612, -0.249529764, -3.474890e-05, 0.107726313, 0.218970656, 0.381412596, -0.529882133, -0.628644109, 0.571398079, 0.299846917, 0.584303737, 0.48202154, 0.328526348, 0.196717009, 0.194961801, 0.152145416, 0.085522361, 0.513142824, 0.0152367353, 0.166441768, 0.332394391, 0.249211237, 0.443366677, -0.280169278, -0.0203848016, -2.45068748E-7, 0.321340501, -4.9151744E-8, 0.237767309, 0.232907727, 0.315274626, 0.427762389, 0.293127537, 0.263794243, 0.675975859, 0.429100394, 0.345662743, -8.69090186E-8, 0.247294366, 0.303160846, 0.615772783, 0.39834857, 0.332067341, -0.412187815, 0.378069043, 0.178953409, 0.25747788, -0.449079722, 0.213058949, 0.569339037, 5.727430e-01, -0.402383476, 0.23406373]> : tensor<64xf32>}> : () -> tensor<64xf32>
    %21 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %22 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %23 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %24 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %26 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %27 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %28 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %29 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %31 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %32 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %33 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %34 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %36 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %37 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %38 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %39 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %41 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %42 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %43 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %44 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %46 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %47 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %48 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %49 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %56 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %57 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %58 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %59 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %61 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %62 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %63 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %66 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %67 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %68 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %69 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %71 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %72 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %73 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %74 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %81 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %82 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %83 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %84 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %86 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %87 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %88 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %89 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %91 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %92 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %93 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %94 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %96 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %97 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %98 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %99 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %106 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %107 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %108 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %109 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %111 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %112 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %113 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %114 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
    %115 = "stablehlo.transpose"(%arg5) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %116 = "stablehlo.convolution"(%115, %15) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<3> : tensor<2x2xi64>, window_strides = array<i64: 2, 2>}> : (tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32>
    %117 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %118 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %119 = "stablehlo.broadcast_in_dim"(%117) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %120 = "stablehlo.subtract"(%116, %119) : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %121 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %122 = "stablehlo.broadcast_in_dim"(%121) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %123 = "stablehlo.add"(%118, %122) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %124 = "stablehlo.rsqrt"(%123) : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %125 = "stablehlo.reshape"(%18) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %126 = "stablehlo.multiply"(%124, %125) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %127 = "stablehlo.broadcast_in_dim"(%126) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %128 = "stablehlo.multiply"(%120, %127) : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %129 = "stablehlo.reshape"(%19) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %130 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
    %131 = "stablehlo.add"(%128, %130) : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %132 = "func.call"(%131) <{callee = @relu}> : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    %133 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %134 = "stablehlo.broadcast_in_dim"(%133) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %135 = "stablehlo.reduce_window"(%132, %134) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
    ^bb0(%arg8: tensor<f32>, %arg9: tensor<f32>):
      %474 = "stablehlo.maximum"(%arg8, %arg9) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%474) : (tensor<f32>) -> ()
    }) : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
    %136 = "stablehlo.convolution"(%135, %20) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %137 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %138 = "stablehlo.broadcast_in_dim"(%22) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %139 = "stablehlo.broadcast_in_dim"(%137) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %140 = "stablehlo.subtract"(%136, %139) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %141 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %142 = "stablehlo.broadcast_in_dim"(%141) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %143 = "stablehlo.add"(%138, %142) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %144 = "stablehlo.rsqrt"(%143) : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %145 = "stablehlo.reshape"(%23) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %146 = "stablehlo.multiply"(%144, %145) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %148 = "stablehlo.multiply"(%140, %147) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %149 = "stablehlo.reshape"(%24) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %150 = "stablehlo.broadcast_in_dim"(%149) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %151 = "stablehlo.add"(%148, %150) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %152 = "func.call"(%151) <{callee = @relu_0}> : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %153 = "stablehlo.convolution"(%152, %25) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %154 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %155 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %156 = "stablehlo.broadcast_in_dim"(%154) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %157 = "stablehlo.subtract"(%153, %156) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %158 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %159 = "stablehlo.broadcast_in_dim"(%158) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %160 = "stablehlo.add"(%155, %159) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %161 = "stablehlo.rsqrt"(%160) : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %162 = "stablehlo.reshape"(%28) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %163 = "stablehlo.multiply"(%161, %162) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %164 = "stablehlo.broadcast_in_dim"(%163) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %165 = "stablehlo.multiply"(%157, %164) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %166 = "stablehlo.reshape"(%29) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %167 = "stablehlo.broadcast_in_dim"(%166) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %168 = "stablehlo.add"(%165, %167) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %169 = "stablehlo.add"(%168, %135) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %170 = "func.call"(%169) <{callee = @relu_0}> : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %171 = "stablehlo.convolution"(%170, %30) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %172 = "stablehlo.broadcast_in_dim"(%31) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %173 = "stablehlo.broadcast_in_dim"(%32) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %174 = "stablehlo.broadcast_in_dim"(%172) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %175 = "stablehlo.subtract"(%171, %174) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %176 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %177 = "stablehlo.broadcast_in_dim"(%176) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %178 = "stablehlo.add"(%173, %177) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %179 = "stablehlo.rsqrt"(%178) : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %180 = "stablehlo.reshape"(%33) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %181 = "stablehlo.multiply"(%179, %180) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %182 = "stablehlo.broadcast_in_dim"(%181) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %183 = "stablehlo.multiply"(%175, %182) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %184 = "stablehlo.reshape"(%34) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %185 = "stablehlo.broadcast_in_dim"(%184) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %186 = "stablehlo.add"(%183, %185) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %187 = "func.call"(%186) <{callee = @relu_0}> : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %188 = "stablehlo.convolution"(%187, %35) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %189 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %190 = "stablehlo.broadcast_in_dim"(%37) <{broadcast_dimensions = array<i64: 3>}> : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %191 = "stablehlo.broadcast_in_dim"(%189) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %192 = "stablehlo.subtract"(%188, %191) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %193 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %194 = "stablehlo.broadcast_in_dim"(%193) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %195 = "stablehlo.add"(%190, %194) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %196 = "stablehlo.rsqrt"(%195) : (tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %197 = "stablehlo.reshape"(%38) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %198 = "stablehlo.multiply"(%196, %197) : (tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    %199 = "stablehlo.broadcast_in_dim"(%198) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %200 = "stablehlo.multiply"(%192, %199) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %201 = "stablehlo.reshape"(%39) : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %202 = "stablehlo.broadcast_in_dim"(%201) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %203 = "stablehlo.add"(%200, %202) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %204 = "stablehlo.add"(%203, %170) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %205 = "func.call"(%204) <{callee = @relu_0}> : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %206 = "stablehlo.convolution"(%205, %40) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, window_strides = array<i64: 2, 2>}> : (tensor<1x56x56x64xf32>, tensor<3x3x64x128xf32>) -> tensor<1x28x28x128xf32>
    %207 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %208 = "stablehlo.broadcast_in_dim"(%42) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %209 = "stablehlo.broadcast_in_dim"(%207) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %210 = "stablehlo.subtract"(%206, %209) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %211 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %212 = "stablehlo.broadcast_in_dim"(%211) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %213 = "stablehlo.add"(%208, %212) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %214 = "stablehlo.rsqrt"(%213) : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %215 = "stablehlo.reshape"(%43) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %216 = "stablehlo.multiply"(%214, %215) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %217 = "stablehlo.broadcast_in_dim"(%216) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %218 = "stablehlo.multiply"(%210, %217) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %219 = "stablehlo.reshape"(%44) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %220 = "stablehlo.broadcast_in_dim"(%219) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %221 = "stablehlo.add"(%218, %220) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %222 = "func.call"(%221) <{callee = @relu_1}> : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %223 = "stablehlo.convolution"(%222, %45) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %224 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %225 = "stablehlo.broadcast_in_dim"(%47) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %226 = "stablehlo.broadcast_in_dim"(%224) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %227 = "stablehlo.subtract"(%223, %226) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %228 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %229 = "stablehlo.broadcast_in_dim"(%228) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %230 = "stablehlo.add"(%225, %229) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %231 = "stablehlo.rsqrt"(%230) : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %232 = "stablehlo.reshape"(%48) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %233 = "stablehlo.multiply"(%231, %232) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %234 = "stablehlo.broadcast_in_dim"(%233) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %235 = "stablehlo.multiply"(%227, %234) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %236 = "stablehlo.reshape"(%49) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %237 = "stablehlo.broadcast_in_dim"(%236) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %238 = "stablehlo.add"(%235, %237) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %239 = "stablehlo.convolution"(%205, %50) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, window_strides = array<i64: 2, 2>}> : (tensor<1x56x56x64xf32>, tensor<1x1x64x128xf32>) -> tensor<1x28x28x128xf32>
    %240 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %241 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %242 = "stablehlo.broadcast_in_dim"(%240) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %243 = "stablehlo.subtract"(%239, %242) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %244 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %245 = "stablehlo.broadcast_in_dim"(%244) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %246 = "stablehlo.add"(%241, %245) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %247 = "stablehlo.rsqrt"(%246) : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %248 = "stablehlo.reshape"(%53) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %249 = "stablehlo.multiply"(%247, %248) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %250 = "stablehlo.broadcast_in_dim"(%249) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %251 = "stablehlo.multiply"(%243, %250) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %252 = "stablehlo.reshape"(%54) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %253 = "stablehlo.broadcast_in_dim"(%252) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %254 = "stablehlo.add"(%251, %253) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %255 = "stablehlo.add"(%238, %254) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %256 = "func.call"(%255) <{callee = @relu_1}> : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %257 = "stablehlo.convolution"(%256, %55) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %258 = "stablehlo.broadcast_in_dim"(%56) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %259 = "stablehlo.broadcast_in_dim"(%57) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %260 = "stablehlo.broadcast_in_dim"(%258) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %261 = "stablehlo.subtract"(%257, %260) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %262 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %263 = "stablehlo.broadcast_in_dim"(%262) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %264 = "stablehlo.add"(%259, %263) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %265 = "stablehlo.rsqrt"(%264) : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %266 = "stablehlo.reshape"(%58) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %267 = "stablehlo.multiply"(%265, %266) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %268 = "stablehlo.broadcast_in_dim"(%267) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %269 = "stablehlo.multiply"(%261, %268) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %270 = "stablehlo.reshape"(%59) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %271 = "stablehlo.broadcast_in_dim"(%270) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %272 = "stablehlo.add"(%269, %271) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %273 = "func.call"(%272) <{callee = @relu_1}> : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %274 = "stablehlo.convolution"(%273, %60) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
    %275 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %276 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64: 3>}> : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %277 = "stablehlo.broadcast_in_dim"(%275) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %278 = "stablehlo.subtract"(%274, %277) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %279 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %280 = "stablehlo.broadcast_in_dim"(%279) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x128xf32>
    %281 = "stablehlo.add"(%276, %280) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %282 = "stablehlo.rsqrt"(%281) : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %283 = "stablehlo.reshape"(%63) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %284 = "stablehlo.multiply"(%282, %283) : (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
    %285 = "stablehlo.broadcast_in_dim"(%284) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %286 = "stablehlo.multiply"(%278, %285) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %287 = "stablehlo.reshape"(%64) : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %288 = "stablehlo.broadcast_in_dim"(%287) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x128xf32>) -> tensor<1x28x28x128xf32>
    %289 = "stablehlo.add"(%286, %288) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %290 = "stablehlo.add"(%289, %256) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %291 = "func.call"(%290) <{callee = @relu_1}> : (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    %292 = "stablehlo.convolution"(%291, %65) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, window_strides = array<i64: 2, 2>}> : (tensor<1x28x28x128xf32>, tensor<3x3x128x256xf32>) -> tensor<1x14x14x256xf32>
    %293 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %294 = "stablehlo.broadcast_in_dim"(%67) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %295 = "stablehlo.broadcast_in_dim"(%293) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %296 = "stablehlo.subtract"(%292, %295) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %297 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %298 = "stablehlo.broadcast_in_dim"(%297) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %299 = "stablehlo.add"(%294, %298) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %300 = "stablehlo.rsqrt"(%299) : (tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %301 = "stablehlo.reshape"(%68) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %302 = "stablehlo.multiply"(%300, %301) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %303 = "stablehlo.broadcast_in_dim"(%302) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %304 = "stablehlo.multiply"(%296, %303) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %305 = "stablehlo.reshape"(%69) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %306 = "stablehlo.broadcast_in_dim"(%305) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %307 = "stablehlo.add"(%304, %306) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %308 = "func.call"(%307) <{callee = @relu_2}> : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %309 = "stablehlo.convolution"(%308, %70) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %310 = "stablehlo.broadcast_in_dim"(%71) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %311 = "stablehlo.broadcast_in_dim"(%72) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %312 = "stablehlo.broadcast_in_dim"(%310) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %313 = "stablehlo.subtract"(%309, %312) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %314 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %315 = "stablehlo.broadcast_in_dim"(%314) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %316 = "stablehlo.add"(%311, %315) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %317 = "stablehlo.rsqrt"(%316) : (tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %318 = "stablehlo.reshape"(%73) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %319 = "stablehlo.multiply"(%317, %318) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %320 = "stablehlo.broadcast_in_dim"(%319) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %321 = "stablehlo.multiply"(%313, %320) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %322 = "stablehlo.reshape"(%74) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %323 = "stablehlo.broadcast_in_dim"(%322) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %324 = "stablehlo.add"(%321, %323) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %325 = "stablehlo.convolution"(%291, %75) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, window_strides = array<i64: 2, 2>}> : (tensor<1x28x28x128xf32>, tensor<1x1x128x256xf32>) -> tensor<1x14x14x256xf32>
    %326 = "stablehlo.broadcast_in_dim"(%76) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %327 = "stablehlo.broadcast_in_dim"(%77) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %328 = "stablehlo.broadcast_in_dim"(%326) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %329 = "stablehlo.subtract"(%325, %328) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %330 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %331 = "stablehlo.broadcast_in_dim"(%330) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %332 = "stablehlo.add"(%327, %331) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %333 = "stablehlo.rsqrt"(%332) : (tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %334 = "stablehlo.reshape"(%78) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %335 = "stablehlo.multiply"(%333, %334) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %336 = "stablehlo.broadcast_in_dim"(%335) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %337 = "stablehlo.multiply"(%329, %336) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %338 = "stablehlo.reshape"(%79) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %339 = "stablehlo.broadcast_in_dim"(%338) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %340 = "stablehlo.add"(%337, %339) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %341 = "stablehlo.add"(%324, %340) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %342 = "func.call"(%341) <{callee = @relu_2}> : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %343 = "stablehlo.convolution"(%342, %80) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %344 = "stablehlo.broadcast_in_dim"(%81) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %345 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %346 = "stablehlo.broadcast_in_dim"(%344) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %347 = "stablehlo.subtract"(%343, %346) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %348 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %349 = "stablehlo.broadcast_in_dim"(%348) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %350 = "stablehlo.add"(%345, %349) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %351 = "stablehlo.rsqrt"(%350) : (tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %352 = "stablehlo.reshape"(%83) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %353 = "stablehlo.multiply"(%351, %352) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %354 = "stablehlo.broadcast_in_dim"(%353) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %355 = "stablehlo.multiply"(%347, %354) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %356 = "stablehlo.reshape"(%84) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %357 = "stablehlo.broadcast_in_dim"(%356) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %358 = "stablehlo.add"(%355, %357) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %359 = "func.call"(%358) <{callee = @relu_2}> : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %360 = "stablehlo.convolution"(%359, %85) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
    %361 = "stablehlo.broadcast_in_dim"(%86) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %362 = "stablehlo.broadcast_in_dim"(%87) <{broadcast_dimensions = array<i64: 3>}> : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %363 = "stablehlo.broadcast_in_dim"(%361) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %364 = "stablehlo.subtract"(%360, %363) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %365 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %366 = "stablehlo.broadcast_in_dim"(%365) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %367 = "stablehlo.add"(%362, %366) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %368 = "stablehlo.rsqrt"(%367) : (tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %369 = "stablehlo.reshape"(%88) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %370 = "stablehlo.multiply"(%368, %369) : (tensor<1x1x1x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x1x1x256xf32>
    %371 = "stablehlo.broadcast_in_dim"(%370) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %372 = "stablehlo.multiply"(%364, %371) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %373 = "stablehlo.reshape"(%89) : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %374 = "stablehlo.broadcast_in_dim"(%373) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x256xf32>) -> tensor<1x14x14x256xf32>
    %375 = "stablehlo.add"(%372, %374) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %376 = "stablehlo.add"(%375, %342) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %377 = "func.call"(%376) <{callee = @relu_2}> : (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    %378 = "stablehlo.convolution"(%377, %90) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, window_strides = array<i64: 2, 2>}> : (tensor<1x14x14x256xf32>, tensor<3x3x256x512xf32>) -> tensor<1x7x7x512xf32>
    %379 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %380 = "stablehlo.broadcast_in_dim"(%92) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %381 = "stablehlo.broadcast_in_dim"(%379) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %382 = "stablehlo.subtract"(%378, %381) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %383 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %384 = "stablehlo.broadcast_in_dim"(%383) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %385 = "stablehlo.add"(%380, %384) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %386 = "stablehlo.rsqrt"(%385) : (tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %387 = "stablehlo.reshape"(%93) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %388 = "stablehlo.multiply"(%386, %387) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %389 = "stablehlo.broadcast_in_dim"(%388) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %390 = "stablehlo.multiply"(%382, %389) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %391 = "stablehlo.reshape"(%94) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %392 = "stablehlo.broadcast_in_dim"(%391) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %393 = "stablehlo.add"(%390, %392) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %394 = "func.call"(%393) <{callee = @relu_3}> : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %395 = "stablehlo.convolution"(%394, %95) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %396 = "stablehlo.broadcast_in_dim"(%96) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %397 = "stablehlo.broadcast_in_dim"(%97) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %398 = "stablehlo.broadcast_in_dim"(%396) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %399 = "stablehlo.subtract"(%395, %398) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %400 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %401 = "stablehlo.broadcast_in_dim"(%400) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %402 = "stablehlo.add"(%397, %401) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %403 = "stablehlo.rsqrt"(%402) : (tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %404 = "stablehlo.reshape"(%98) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %405 = "stablehlo.multiply"(%403, %404) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %406 = "stablehlo.broadcast_in_dim"(%405) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %407 = "stablehlo.multiply"(%399, %406) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %408 = "stablehlo.reshape"(%99) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %409 = "stablehlo.broadcast_in_dim"(%408) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %410 = "stablehlo.add"(%407, %409) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %411 = "stablehlo.convolution"(%377, %100) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, window_strides = array<i64: 2, 2>}> : (tensor<1x14x14x256xf32>, tensor<1x1x256x512xf32>) -> tensor<1x7x7x512xf32>
    %412 = "stablehlo.broadcast_in_dim"(%101) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %413 = "stablehlo.broadcast_in_dim"(%102) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %414 = "stablehlo.broadcast_in_dim"(%412) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %415 = "stablehlo.subtract"(%411, %414) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %416 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %417 = "stablehlo.broadcast_in_dim"(%416) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %418 = "stablehlo.add"(%413, %417) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %419 = "stablehlo.rsqrt"(%418) : (tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %420 = "stablehlo.reshape"(%103) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %421 = "stablehlo.multiply"(%419, %420) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %422 = "stablehlo.broadcast_in_dim"(%421) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %423 = "stablehlo.multiply"(%415, %422) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %424 = "stablehlo.reshape"(%104) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %425 = "stablehlo.broadcast_in_dim"(%424) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %426 = "stablehlo.add"(%423, %425) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %427 = "stablehlo.add"(%410, %426) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %428 = "func.call"(%427) <{callee = @relu_3}> : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %429 = "stablehlo.convolution"(%428, %105) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %430 = "stablehlo.broadcast_in_dim"(%106) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %431 = "stablehlo.broadcast_in_dim"(%107) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %432 = "stablehlo.broadcast_in_dim"(%430) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %433 = "stablehlo.subtract"(%429, %432) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %434 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %435 = "stablehlo.broadcast_in_dim"(%434) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %436 = "stablehlo.add"(%431, %435) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %437 = "stablehlo.rsqrt"(%436) : (tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %438 = "stablehlo.reshape"(%108) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %439 = "stablehlo.multiply"(%437, %438) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %440 = "stablehlo.broadcast_in_dim"(%439) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %441 = "stablehlo.multiply"(%433, %440) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %442 = "stablehlo.reshape"(%109) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %443 = "stablehlo.broadcast_in_dim"(%442) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %444 = "stablehlo.add"(%441, %443) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %445 = "func.call"(%444) <{callee = @relu_3}> : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %446 = "stablehlo.convolution"(%445, %110) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>}> : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
    %447 = "stablehlo.broadcast_in_dim"(%111) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %448 = "stablehlo.broadcast_in_dim"(%112) <{broadcast_dimensions = array<i64: 3>}> : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %449 = "stablehlo.broadcast_in_dim"(%447) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %450 = "stablehlo.subtract"(%446, %449) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %451 = "stablehlo.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
    %452 = "stablehlo.broadcast_in_dim"(%451) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %453 = "stablehlo.add"(%448, %452) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %454 = "stablehlo.rsqrt"(%453) : (tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %455 = "stablehlo.reshape"(%113) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %456 = "stablehlo.multiply"(%454, %455) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %457 = "stablehlo.broadcast_in_dim"(%456) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %458 = "stablehlo.multiply"(%450, %457) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %459 = "stablehlo.reshape"(%114) : (tensor<512xf32>) -> tensor<1x1x1x512xf32>
    %460 = "stablehlo.broadcast_in_dim"(%459) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x512xf32>) -> tensor<1x7x7x512xf32>
    %461 = "stablehlo.add"(%458, %460) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %462 = "stablehlo.add"(%461, %428) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %463 = "func.call"(%462) <{callee = @relu_3}> : (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    %464 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %465 = "stablehlo.broadcast_in_dim"(%464) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %466 = "stablehlo.reduce_window"(%463, %465) <{window_dimensions = array<i64: 1, 7, 7, 1>, window_strides = array<i64: 1, 7, 7, 1>}> ({
    ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
      %473 = "stablehlo.add"(%arg6, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%473) : (tensor<f32>) -> ()
    }) : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x1x1x512xf32>
    %467 = "stablehlo.constant"() <{value = dense<49> : tensor<i32>}> : () -> tensor<i32>
    %468 = "stablehlo.convert"(%467) : (tensor<i32>) -> tensor<f32>
    %469 = "stablehlo.broadcast_in_dim"(%468) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1x512xf32>
    %470 = "stablehlo.divide"(%466, %469) : (tensor<1x1x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x1x1x512xf32>
    %471 = "stablehlo.transpose"(%470) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x1x1x512xf32>) -> tensor<1x512x1x1xf32>
    %472 = "stablehlo.transpose"(%463) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x7x7x512xf32>) -> tensor<1x512x7x7xf32>
    "func.return"(%472, %471) : (tensor<1x512x7x7xf32>, tensor<1x512x1x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "relu", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<1x112x112x64xf32>):
    %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %13 = "stablehlo.broadcast_in_dim"(%12) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x112x112x64xf32>
    %14 = "stablehlo.maximum"(%arg4, %13) : (tensor<1x112x112x64xf32>, tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
    "func.return"(%14) : (tensor<1x112x112x64xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "relu_0", sym_visibility = "private"}> ({
  ^bb0(%arg3: tensor<1x56x56x64xf32>):
    %9 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %11 = "stablehlo.maximum"(%arg3, %10) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    "func.return"(%11) : (tensor<1x56x56x64xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "relu_1", sym_visibility = "private"}> ({
  ^bb0(%arg2: tensor<1x28x28x128xf32>):
    %6 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x28x28x128xf32>
    %8 = "stablehlo.maximum"(%arg2, %7) : (tensor<1x28x28x128xf32>, tensor<1x28x28x128xf32>) -> tensor<1x28x28x128xf32>
    "func.return"(%8) : (tensor<1x28x28x128xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "relu_2", sym_visibility = "private"}> ({
  ^bb0(%arg1: tensor<1x14x14x256xf32>):
    %3 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %4 = "stablehlo.broadcast_in_dim"(%3) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x14x14x256xf32>
    %5 = "stablehlo.maximum"(%arg1, %4) : (tensor<1x14x14x256xf32>, tensor<1x14x14x256xf32>) -> tensor<1x14x14x256xf32>
    "func.return"(%5) : (tensor<1x14x14x256xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "relu_3", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<1x7x7x512xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x7x7x512xf32>
    %2 = "stablehlo.maximum"(%arg0, %1) : (tensor<1x7x7x512xf32>, tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
    "func.return"(%2) : (tensor<1x7x7x512xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

