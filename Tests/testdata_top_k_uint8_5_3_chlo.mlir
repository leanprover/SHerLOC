"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<5x2xui8>, tensor<5x2xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<5x3xui8>
    %4:2 = "func.call"() <{callee = @expected}> : () -> (tensor<5x2xui8>, tensor<5x2xi32>)
    %5:2 = "chlo.top_k"(%3) <{k = 2 : i64}> : (tensor<5x3xui8>) -> (tensor<5x2xui8>, tensor<5x2xi32>)
    "stablehlo.custom_call"(%5#0, %4#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xui8>, tensor<5x2xui8>) -> ()
    "stablehlo.custom_call"(%5#1, %4#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    "func.return"(%5#0, %5#1) : (tensor<5x2xui8>, tensor<5x2xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x3xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0, 1], [3, 5, 1], [0, 6, 1], [0, 2, 2], [4, 1, 1]]> : tensor<5x3xui8>}> : () -> tensor<5x3xui8>
    "func.return"(%2) : (tensor<5x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x2xui8>, tensor<5x2xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 0], [5, 3], [6, 1], [2, 2], [4, 1]]> : tensor<5x2xui8>}> : () -> tensor<5x2xui8>
    %1 = "stablehlo.constant"() <{value = dense<[[2, 0], [1, 0], [1, 2], [1, 2], [0, 1]]> : tensor<5x2xi32>}> : () -> tensor<5x2xi32>
    "func.return"(%0, %1) : (tensor<5x2xui8>, tensor<5x2xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

