"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf64>
    %6 = "stablehlo.select"(%4#0, %4#2, %4#1) : (tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    "func.return"(%6) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %2 = "stablehlo.constant"() <{value = dense<[[1.6545636564404331, -5.4821347976364159, -2.0086517908148904], [0.69937796660157336, -3.0890795754926761, 0.16769550090612578]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    %3 = "stablehlo.constant"() <{value = dense<[[-1.7115590272350207, -2.9520499628953853, -0.27612026716671112], [2.7204079321798487, 3.1137611728964445, -2.6623861275903633]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1, %2, %3) : (tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1.7115590272350207, -2.9520499628953853, -0.27612026716671112], [2.7204079321798487, 3.1137611728964445, -2.6623861275903633]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%0) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

