"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xi32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xi32>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x4xi32>, tensor<3x4xi32>) -> ()
    "func.return"(%2) : (tensor<3x4xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1, 3, 1, 3], [2, 0, 3, -3], [0, 0, 1, 0]]> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
    "func.return"(%1) : (tensor<3x4xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1, 3, 1, 3], [2, 0, 3, -3], [0, 0, 1, 0]]> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
    "func.return"(%0) : (tensor<3x4xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

