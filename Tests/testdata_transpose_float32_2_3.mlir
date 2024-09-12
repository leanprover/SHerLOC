"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xf32>
    %4 = "stablehlo.transpose"(%2) <{permutation = array<i64: 1, 0>}> : (tensor<2x3xf32>) -> tensor<3x2xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
    "func.return"(%4) : (tensor<3x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.15674615, 0.146370798, 0.875410914], [-0.362745672, -0.542615533, -2.79061985]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%1) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.15674615, -0.362745672], [0.146370798, -0.542615533], [0.875410914, -2.79061985]]> : tensor<3x2xf32>}> : () -> tensor<3x2xf32>
    "func.return"(%0) : (tensor<3x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

