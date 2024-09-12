"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4xi16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4xi16>
    %4 = "stablehlo.popcnt"(%2) : (tensor<4xi16>) -> tensor<4xi16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4xi16>, tensor<4xi16>) -> ()
    "func.return"(%4) : (tensor<4xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-1, -2, 0, 1]> : tensor<4xi16>}> : () -> tensor<4xi16>
    "func.return"(%1) : (tensor<4xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[16, 15, 0, 1]> : tensor<4xi16>}> : () -> tensor<4xi16>
    "func.return"(%0) : (tensor<4xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

