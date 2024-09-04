"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %6 = "func.call"(%5) <{callee = @_cumulative_reduction}> : (tensor<3xf32>) -> tensor<3xf32>
    "func.return"(%6) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<3xf32>) -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_cumulative_reduction", sym_visibility = "private"}> ({
  ^bb0(%arg3: tensor<3xf32>):
    %4 = "func.call"(%arg3) <{callee = @cumsum}> : (tensor<3xf32>) -> tensor<3xf32>
    "func.return"(%4) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<3xf32>) -> tensor<3xf32>, sym_name = "cumsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<3xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[2, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 3>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
    }) : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
    "func.return"(%2) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

