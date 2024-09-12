"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x1xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x1xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<1x1xf32>
    %4 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<2x1xf32>, tensor<f32>) -> tensor<1x1xf32>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x1xf32>, tensor<1x1xf32>) -> ()
    "func.return"(%6) : (tensor<1x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x1xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.36937273], [-2.20683861]]> : tensor<2x1xf32>}> : () -> tensor<2x1xf32>
    "func.return"(%1) : (tensor<2x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x1xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-1.36937273> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    "func.return"(%0) : (tensor<1x1xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

