"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xbf16>, tensor<2xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xbf16>
    %5 = "stablehlo.remainder"(%3#0, %3#1) : (tensor<2xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xbf16>, tensor<2xbf16>) -> ()
    "func.return"(%5) : (tensor<2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xbf16>, tensor<2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[5.468750e+00, -1.320310e+00]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[-3.632810e-01, 4.433590e-01]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    "func.return"(%1, %2) : (tensor<2xbf16>, tensor<2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.953130e-02, -4.335940e-01]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    "func.return"(%0) : (tensor<2xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

