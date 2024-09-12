"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x3xf32>
    %5 = "stablehlo.broadcast_in_dim"(%3#0) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %6 = "stablehlo.remainder"(%5, %3#1) : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> ()
    "func.return"(%6) : (tensor<2x4x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-0.616980791, 0.706572294, -4.73949623]], [[1.67496932, -3.13438964, 1.29330683]]]> : tensor<2x1x3xf32>}> : () -> tensor<2x1x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[-0.0259232726, 5.1987915, 7.33156443], [3.09692621, 0.534493804, 2.14104652], [-4.89970493, -0.817309558, -1.55429268], [4.97919846, -4.20424652, -0.378917187]], [[2.69760561, -1.81906879, 0.697872698], [-2.84189224, 2.68079877, 0.158963606], [-1.83596051, -1.92423916, -0.504996121], [-1.42828655, -3.60045314, 4.78329611]]]> : tensor<2x4x3xf32>}> : () -> tensor<2x4x3xf32>
    "func.return"(%1, %2) : (tensor<2x1x3xf32>, tensor<2x4x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-0.0207455214, 0.706572294, -4.73949623], [-0.616980791, 0.17207849, -0.457403183], [-0.616980791, 0.706572294, -0.0766181945], [-0.616980791, 0.706572294, -0.192489982]], [[1.67496932, -1.31532085, 0.595434129], [1.67496932, -0.45359087, 0.0215979815], [1.67496932, -1.21015048, 0.283314586], [0.246682763, -3.13438964, 1.29330683]]]> : tensor<2x4x3xf32>}> : () -> tensor<2x4x3xf32>
    "func.return"(%0) : (tensor<2x4x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

