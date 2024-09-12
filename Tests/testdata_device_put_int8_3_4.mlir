"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xi8>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xi8>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x4xi8>, tensor<3x4xi8>) -> ()
    "func.return"(%2) : (tensor<3x4xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4, 4, -2, 4], [-1, 5, 0, 0], [-3, -3, 4, -1]]> : tensor<3x4xi8>}> : () -> tensor<3x4xi8>
    "func.return"(%1) : (tensor<3x4xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4, 4, -2, 4], [-1, 5, 0, 0], [-3, -3, 4, -1]]> : tensor<3x4xi8>}> : () -> tensor<3x4xi8>
    "func.return"(%0) : (tensor<3x4xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

