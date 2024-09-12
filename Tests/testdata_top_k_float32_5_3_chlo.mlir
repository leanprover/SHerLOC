"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<5x2xf32>, tensor<5x2xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<5x3xf32>
    %4:2 = "func.call"() <{callee = @expected}> : () -> (tensor<5x2xf32>, tensor<5x2xi32>)
    %5:2 = "chlo.top_k"(%3) <{k = 2 : i64}> : (tensor<5x3xf32>) -> (tensor<5x2xf32>, tensor<5x2xi32>)
    "stablehlo.custom_call"(%5#0, %4#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xf32>, tensor<5x2xf32>) -> ()
    "stablehlo.custom_call"(%5#1, %4#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    "func.return"(%5#0, %5#1) : (tensor<5x2xf32>, tensor<5x2xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[1.87676096, -1.96814215, 0.339208782], [-1.78530681, 1.6039784, -1.19955933], [0.210248947, 2.594690e+00, -0.506774485], [2.07919931, -3.76786542, -2.9934845], [0.578959584, -0.907130658, -0.925940394]]> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
    "func.return"(%2) : (tensor<5x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x2xf32>, tensor<5x2xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.87676096, 0.339208782], [1.6039784, -1.19955933], [2.594690e+00, 0.210248947], [2.07919931, -2.9934845], [0.578959584, -0.907130658]]> : tensor<5x2xf32>}> : () -> tensor<5x2xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[0, 2], [1, 2], [1, 0], [0, 2], [0, 1]]> : tensor<5x2xi32>}> : () -> tensor<5x2xi32>
    "func.return"(%0, %1) : (tensor<5x2xf32>, tensor<5x2xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

