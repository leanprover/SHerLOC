"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xi16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xi16>
    %4 = "stablehlo.transpose"(%2) <{permutation = array<i64: 1, 0>}> : (tensor<2x3xi16>) -> tensor<3x2xi16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x2xi16>, tensor<3x2xi16>) -> ()
    "func.return"(%4) : (tensor<3x2xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4, -1, 0], [4, 6, -1]]> : tensor<2x3xi16>}> : () -> tensor<2x3xi16>
    "func.return"(%1) : (tensor<2x3xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4, 4], [-1, 6], [0, -1]]> : tensor<3x2xi16>}> : () -> tensor<3x2xi16>
    "func.return"(%0) : (tensor<3x2xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

