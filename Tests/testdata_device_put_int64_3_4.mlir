"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xi64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xi64>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x4xi64>, tensor<3x4xi64>) -> ()
    "func.return"(%2) : (tensor<3x4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2, -1, 2, 4], [0, 1, -1, -5], [0, 2, 0, 0]]> : tensor<3x4xi64>}> : () -> tensor<3x4xi64>
    "func.return"(%1) : (tensor<3x4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2, -1, 2, 4], [0, 1, -1, -5], [0, 2, 0, 0]]> : tensor<3x4xi64>}> : () -> tensor<3x4xi64>
    "func.return"(%0) : (tensor<3x4xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

