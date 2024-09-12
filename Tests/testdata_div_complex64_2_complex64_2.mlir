"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xcomplex<f32>>
    %5 = "stablehlo.divide"(%3#0, %3#1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<2xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[(4.82876539,1.14962423), (0.816298723,-3.73786092)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %2 = "stablehlo.constant"() <{value = dense<[(-6.938800e-01,-4.50840855), (-1.38481331,3.32134724)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(-0.410124958,1.00793612), (-1.04603434,0.190363556)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    "func.return"(%0) : (tensor<2xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

