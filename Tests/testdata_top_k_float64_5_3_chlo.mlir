"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<5x2xf64>, tensor<5x2xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<5x3xf64>
    %4:2 = "func.call"() <{callee = @expected}> : () -> (tensor<5x2xf64>, tensor<5x2xi32>)
    %5:2 = "chlo.top_k"(%3) <{k = 2 : i64}> : (tensor<5x3xf64>) -> (tensor<5x2xf64>, tensor<5x2xi32>)
    "stablehlo.custom_call"(%5#0, %4#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xf64>, tensor<5x2xf64>) -> ()
    "stablehlo.custom_call"(%5#1, %4#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    "func.return"(%5#0, %5#1) : (tensor<5x2xf64>, tensor<5x2xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[1.0343520378830138, -0.72005861577891961, 0.8116893130582965], [1.607892529832593, 0.98557547257388722, 2.4222800833698059], [-1.9816222607811027, -1.7198411085428242, -6.5269585694018968], [-2.0504014852139987, 0.91583800697560958, 3.4342856142046685], [-3.0949193319215005, -3.6759015659403387, 6.5702934608478234]]> : tensor<5x3xf64>}> : () -> tensor<5x3xf64>
    "func.return"(%2) : (tensor<5x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x2xf64>, tensor<5x2xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.0343520378830138, 0.8116893130582965], [2.4222800833698059, 1.607892529832593], [-1.7198411085428242, -1.9816222607811027], [3.4342856142046685, 0.91583800697560958], [6.5702934608478234, -3.0949193319215005]]> : tensor<5x2xf64>}> : () -> tensor<5x2xf64>
    %1 = "stablehlo.constant"() <{value = dense<[[0, 2], [2, 0], [1, 0], [2, 1], [2, 0]]> : tensor<5x2xi32>}> : () -> tensor<5x2xi32>
    "func.return"(%0, %1) : (tensor<5x2xf64>, tensor<5x2xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

