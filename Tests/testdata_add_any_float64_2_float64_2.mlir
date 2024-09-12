"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xf64>, tensor<2xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xf64>
    %5 = "stablehlo.add"(%3#0, %3#1) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xf64>, tensor<2xf64>) -> ()
    "func.return"(%5) : (tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xf64>, tensor<2xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-0.54167594873728864, 3.6439287424073785]> : tensor<2xf64>}> : () -> tensor<2xf64>
    %2 = "stablehlo.constant"() <{value = dense<[2.4805639646388817, 3.4586733000966037]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%1, %2) : (tensor<2xf64>, tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.9388880159015931, 7.1026020425039817]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%0) : (tensor<2xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

