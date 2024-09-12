"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui32>
    %4 = "stablehlo.bitcast_convert"(%2) : (tensor<2x3xf32>) -> tensor<2x3xui32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui32>, tensor<2x3xui32>) -> ()
    "func.return"(%4) : (tensor<2x3xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.88355398, -3.74272418, 2.57351112], [-5.73217392, -2.79588485, -1.04740441]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%1) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3224931366, 3228534987, 1076147304], [3233246712, 3224563655, 3213234521]]> : tensor<2x3xui32>}> : () -> tensor<2x3xui32>
    "func.return"(%0) : (tensor<2x3xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

