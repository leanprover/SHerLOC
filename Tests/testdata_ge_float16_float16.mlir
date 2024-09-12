"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<f16>, tensor<f16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<i1>
    %5 = "stablehlo.compare"(%3#0, %3#1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i1>, tensor<i1>) -> ()
    "func.return"(%5) : (tensor<i1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<f16>, tensor<f16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<7.353520e-01> : tensor<f16>}> : () -> tensor<f16>
    %2 = "stablehlo.constant"() <{value = dense<7.152340e+00> : tensor<f16>}> : () -> tensor<f16>
    "func.return"(%1, %2) : (tensor<f16>, tensor<f16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    "func.return"(%0) : (tensor<i1>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

