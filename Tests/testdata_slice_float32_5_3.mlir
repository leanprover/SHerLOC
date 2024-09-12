"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x3xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x2xf32>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 5, 3>, start_indices = array<i64: 1, 1>, strides = array<i64: 2, 1>}> : (tensor<5x3xf32>) -> tensor<2x2xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "func.return"(%4) : (tensor<2x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.551178575, 2.42043257, 6.3848033], [-1.56378138, -0.768643915, -3.33495092], [7.55508327, -0.759385645, -0.365653396], [-2.15069771, -1.417720e+00, -3.05872202], [-3.96960521, 0.531166673, -2.26626372]]> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
    "func.return"(%1) : (tensor<5x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-0.768643915, -3.33495092], [-1.417720e+00, -3.05872202]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    "func.return"(%0) : (tensor<2x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

