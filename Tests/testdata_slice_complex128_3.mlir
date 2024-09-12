"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<1xcomplex<f64>>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xcomplex<f64>>) -> tensor<1xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1xcomplex<f64>>, tensor<1xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<1xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[(4.3319333664393156,1.7674270116019666), (-2.8423782718122377,0.71376902446208579), (-2.1525327045932272,-2.8886484621089434)]> : tensor<3xcomplex<f64>>}> : () -> tensor<3xcomplex<f64>>
    "func.return"(%1) : (tensor<3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(-2.8423782718122377,0.71376902446208579)> : tensor<1xcomplex<f64>>}> : () -> tensor<1xcomplex<f64>>
    "func.return"(%0) : (tensor<1xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

