"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<1x2xi8>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2xi8>
    %4 = "stablehlo.reshape"(%2) : (tensor<1x2xi8>) -> tensor<2xi8>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2xi8>, tensor<2xi8>) -> ()
    "func.return"(%4) : (tensor<2xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2, 3]]> : tensor<1x2xi8>}> : () -> tensor<1x2xi8>
    "func.return"(%1) : (tensor<1x2xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, 3]> : tensor<2xi8>}> : () -> tensor<2xi8>
    "func.return"(%0) : (tensor<2xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

