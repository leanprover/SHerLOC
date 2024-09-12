"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<1x2xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2xcomplex<f64>>
    %4 = "stablehlo.reshape"(%2) : (tensor<1x2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(0.95246291894714041,-0.58390554129361716), (2.0702535372094326,5.8459422374991075)]]> : tensor<1x2xcomplex<f64>>}> : () -> tensor<1x2xcomplex<f64>>
    "func.return"(%1) : (tensor<1x2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0.95246291894714041,-0.58390554129361716), (2.0702535372094326,5.8459422374991075)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    "func.return"(%0) : (tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

