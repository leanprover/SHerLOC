"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf32>
    %6 = "stablehlo.select"(%4#0, %4#2, %4#1) : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    "func.return"(%6) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %2 = "stablehlo.constant"() <{value = dense<[[-0.477684617, 1.12567687, -6.458900e+00], [0.738749385, -4.38916922, -4.62747955]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %3 = "stablehlo.constant"() <{value = dense<[[3.92670202, 1.27711701, 4.59667206], [-3.72463632, -1.26884151, -4.21353149]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%1, %2, %3) : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3.92670202, 1.27711701, 4.59667206], [-3.72463632, -1.26884151, -4.21353149]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%0) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

