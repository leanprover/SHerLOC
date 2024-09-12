"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<12x12xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<6x6xf32>
    %4 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %6 = "stablehlo.reduce_window"(%2, %5) <{padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<12x12xf32>, tensor<f32>) -> tensor<6x6xf32>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6x6xf32>, tensor<6x6xf32>) -> ()
    "func.return"(%6) : (tensor<6x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<12x12xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x43A9C6C05841E93F0ED717C0200D733F5428EFBF15CB09C0089F9B3F4E52C1BFBADA7EBEA4064EC0C73A3AC0773F80BEBFBBB83EE21C494053F9B63F25B7A4BE68FE9AC0D16D8DBE87D08DBFE33DECBFF768E93FCF5BC03FC3EE97BFA24E4E3EE182B3BF9F7D404087C1BDC0C599963FE14155BFC372E13F556AA3C07B5D47C0463C66C0F63E7440083A7B40DAE02DBF853735BDBC8409BF6EDC09BF4EE989BF73DFB440254FF13F407EF03F52DB483FF7E96A400D1B76C01E66A53EE5F8B1BF9D3B2340DE5362C0F4162340D7432740ED5E1DC00D8CFFBF39A97BBE10BDA0C0664B563F4672064029AD463F2F506CC08793DBC0CAD0F6BF74BDE440D90C1AC0DC57F9BF795019C01A4492C0918869C0EF6519C0D32B77C0E91F494098BB9BBF7A30BD40A00447C0EE3A78C0D2F385407F05AA3F12EF3EBFA3EA24400F02EA3F75DE46C06C570BC0860F80BF1160CFBFD9E6F2BFA3DD8640B31B44BF8239B0C07E734DC008FF0EC07A052F40214159C00187D8BF29A62540138989406E9D7ABF34FC5FC0525AE5C01D890E3F2D8FC03F77F7E7BE4623B5C02C18624020D4204089E4D83F082161C0BCCC2B3FEE7ACC3FD09153400B7633C07F3DBB3FB93031BF709D3040B0E771401ABED13F1FD6213FF4B0813F851B7F40CB40BFBF126D28C0FB8931C1A17B5240F1095F3F11DB5F40141E3B40C8C912C02D6083409FAD2BBF225AF2BDE4FC40402FD970C0C9DD603DC9979DC00DBDDC3F3572B33FE05078C082E29DBF8B57E73FA02795BE859B70C0E4C735C00DA8B140C5C6D440443CC940"> : tensor<12x12xf32>}> : () -> tensor<12x12xf32>
    "func.return"(%1) : (tensor<12x12xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-6.15031433, -12.6136236, -13.2118111, -13.4927034, -0.0941996574, -0.899619221], [-3.88216138, 1.1528542, 0.536762357, -9.91701316, 8.00639725, -0.737472773], [-1.15554476, 7.13168097, -10.4562073, -13.7506924, -5.68744373, -3.6109972], [-9.6355133, -6.23944283, -2.13139272, 6.68631744, -2.24245691, 2.96701908], [-1.508190e+01, 12.4310646, 10.3327332, 14.340683, 1.4868722, -5.56558418], [-8.73448085, 3.58121395, 5.01820135, -3.57459927, 8.49454593, 9.22951316]]> : tensor<6x6xf32>}> : () -> tensor<6x6xf32>
    "func.return"(%0) : (tensor<6x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

