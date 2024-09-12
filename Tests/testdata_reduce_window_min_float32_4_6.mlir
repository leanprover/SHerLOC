"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf32>
    %4 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    "func.return"(%6) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-0.587396801, 1.41679704, 1.57286632, 2.75978231, -1.49650061, -2.30756736], [3.0710969, 4.19341516, 2.3716979, -3.34889412, 2.2374773, -1.73396564], [1.05131531, -0.597165704, 5.87361765, -0.149147883, -1.62176561, 0.108428128], [-2.76762557, 2.69764709, -1.18576527, -1.25375938, -2.87609529, 2.79383397]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    "func.return"(%1) : (tensor<4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-0.587396801, 1.41679704, -3.34889412, -3.34889412, -2.30756736], [-0.597165704, -0.597165704, -3.34889412, -3.34889412, -1.73396564], [-2.76762557, -1.18576527, -1.25375938, -2.87609529, -2.87609529]]> : tensor<3x5xf32>}> : () -> tensor<3x5xf32>
    "func.return"(%0) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

