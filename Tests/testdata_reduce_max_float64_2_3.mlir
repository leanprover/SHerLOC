"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xf64>
    %4 = "stablehlo.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%6) : (tensor<f64>) -> ()
    }) : (tensor<2x3xf64>, tensor<f64>) -> tensor<3xf64>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf64>, tensor<3xf64>) -> ()
    "func.return"(%5) : (tensor<3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.26309286187879577, -1.9087842265701214, -0.64194290673004495], [2.3193875620984707, -5.559375664459191, 3.3056404880737893]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[2.3193875620984707, -1.9087842265701214, 3.3056404880737893]> : tensor<3xf64>}> : () -> tensor<3xf64>
    "func.return"(%0) : (tensor<3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

