"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf32>
    %6 = "stablehlo.broadcast_in_dim"(%4#0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<2x3xf32>
    %7 = "stablehlo.clamp"(%6, %4#1, %4#2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    "stablehlo.custom_call"(%7, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    "func.return"(%7) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3.66605306, 1.330880e+01, 4.89149714], [1.76893127, -6.03403234, 0.186552331]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[-4.00151539, -1.59384251, 3.47588778], [3.14183354, 0.30163914, -1.07248604]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %3 = "stablehlo.constant"() <{value = dense<-0.638752341> : tensor<f32>}> : () -> tensor<f32>
    "func.return"(%3, %1, %2) : (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.00151539, -1.59384251, 3.47588778], [1.76893127, -0.638752341, -1.07248604]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%0) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

