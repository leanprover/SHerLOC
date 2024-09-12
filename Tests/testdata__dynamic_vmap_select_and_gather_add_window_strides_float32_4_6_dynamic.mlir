"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4x6xf32>, tensor<?x4x6xf32>) -> tensor<?x2x2xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4x6xf32>, %arg2: tensor<?x4x6xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %2:2 = "stablehlo.reduce_window"(%arg2, %arg1, %0, %1) <{window_dimensions = array<i64: 1, 2, 2>, window_strides = array<i64: 1, 2, 3>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>):
      %3 = "stablehlo.compare"(%arg3, %arg5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = "stablehlo.select"(%3, %arg3, %arg5) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %5 = "stablehlo.select"(%3, %arg4, %arg6) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
    }) : (tensor<?x4x6xf32>, tensor<?x4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<?x2x2xf32>, tensor<?x2x2xf32>)
    "func.return"(%2#1) : (tensor<?x2x2xf32>) -> ()
  }) : () -> ()
}) : () -> ()

