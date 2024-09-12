"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xf64>, tensor<2xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xf64>
    %5 = "stablehlo.divide"(%3#0, %3#1) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xf64>, tensor<2xf64>) -> ()
    "func.return"(%5) : (tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xf64>, tensor<2xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-0.18840514160007904, 0.49408547321691121]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %2 = "stablehlo.constant"() <{value = dense<[1.864675550364308, 4.998694464524486]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%1, %2) : (tensor<2xf64>, tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-0.10103910117943558, 0.098842903226715295]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%0) : (tensor<2xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

