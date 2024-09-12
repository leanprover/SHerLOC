"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<f64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<f64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<f64>
    %4 = "stablehlo.reduce_precision"(%2) <{exponent_bits = 5 : i32, mantissa_bits = 10 : i32}> : (tensor<f64>) -> tensor<f64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<f64>, tensor<f64>) -> ()
    "func.return"(%4) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<1.2006864362520839> : tensor<f64>}> : () -> tensor<f64>
    "func.return"(%1) : (tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<f64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<1.201171875> : tensor<f64>}> : () -> tensor<f64>
    "func.return"(%0) : (tensor<f64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

