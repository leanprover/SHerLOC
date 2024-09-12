"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xf64>
    %4 = "stablehlo.reshape"(%2) : (tensor<2x3xf64>) -> tensor<3x2xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xf64>, tensor<3x2xf64>) -> ()
    "func.return"(%4) : (tensor<3x2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.4769515279267078, -3.9417650234036818, 3.7265509545172635], [-3.5866178507101436, -5.8366036882496086, -1.3450993943329561]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.4769515279267078, -3.9417650234036818], [3.7265509545172635, -3.5866178507101436], [-5.8366036882496086, -1.3450993943329561]]> : tensor<3x2xf64>}> : () -> tensor<3x2xf64>
    "func.return"(%0) : (tensor<3x2xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

