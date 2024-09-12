"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xf32>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xf32>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %6 = "stablehlo.compare"(%arg0, %5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %8 = "stablehlo.select"(%6, %7, %arg0) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %9 = "stablehlo.compare"(%arg0, %arg0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %11 = "stablehlo.select"(%9, %10, %8) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %13 = "stablehlo.compare"(%arg1, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %15 = "stablehlo.select"(%13, %14, %arg1) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %16 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %17 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
      %18 = "stablehlo.select"(%16, %17, %15) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %19 = "stablehlo.compare"(%11, %18) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%19) : (tensor<i1>) -> ()
    }) : (tensor<5x7xf32>) -> tensor<5x7xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xf32>, tensor<5x7xf32>) -> ()
    "func.return"(%4) : (tensor<5x7xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.07987475, 2.40121102, 0.609614312, 2.11366987, -3.75532937, -1.70798647, -1.60148156], [1.85626411, -0.232852742, 3.18949652, -2.20193195, -0.26601395, -3.44230628, -1.21295929], [-3.16205668, 0.570658684, -1.29143703, -1.20841086, 4.32051897, 1.61176097, -3.21153545], [0.805976331, 6.68937445, -1.94188154, -1.46865034, 0.651393652, 3.10745668, 9.86636352], [-0.469363391, 1.07057512, 4.19001722, 3.66513085, 2.18365979, -0.231317356, -0.3270244]]> : tensor<5x7xf32>}> : () -> tensor<5x7xf32>
    "func.return"(%1) : (tensor<5x7xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3.16205668, -0.232852742, -1.94188154, -2.20193195, -3.75532937, -3.44230628, -3.21153545], [-1.07987475, 0.570658684, -1.29143703, -1.46865034, -0.26601395, -1.70798647, -1.60148156], [-0.469363391, 1.07057512, 0.609614312, -1.20841086, 0.651393652, -0.231317356, -1.21295929], [0.805976331, 2.40121102, 3.18949652, 2.11366987, 2.18365979, 1.61176097, -0.3270244], [1.85626411, 6.68937445, 4.19001722, 3.66513085, 4.32051897, 3.10745668, 9.86636352]]> : tensor<5x7xf32>}> : () -> tensor<5x7xf32>
    "func.return"(%0) : (tensor<5x7xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

