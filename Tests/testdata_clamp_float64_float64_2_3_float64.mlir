"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<f64>, tensor<2x3xf64>, tensor<f64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf64>
    %6 = "stablehlo.broadcast_in_dim"(%4#0) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<2x3xf64>
    %7 = "stablehlo.broadcast_in_dim"(%4#2) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<2x3xf64>
    %8 = "stablehlo.clamp"(%6, %4#1, %7) : (tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    "stablehlo.custom_call"(%8, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    "func.return"(%8) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<f64>, tensor<2x3xf64>, tensor<f64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1.3647748627855825, 0.55441559336763024, -2.355556198180957], [2.2665459589395178, 0.58817989765507428, 3.7514669634211639]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<2.6354240164456249> : tensor<f64>}> : () -> tensor<f64>
    %3 = "stablehlo.constant"() <{value = dense<4.0043262680460456> : tensor<f64>}> : () -> tensor<f64>
    "func.return"(%2, %1, %3) : (tensor<f64>, tensor<2x3xf64>, tensor<f64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[2.6354240164456249, 2.6354240164456249, 2.6354240164456249], [2.6354240164456249, 2.6354240164456249, 3.7514669634211639]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%0) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

