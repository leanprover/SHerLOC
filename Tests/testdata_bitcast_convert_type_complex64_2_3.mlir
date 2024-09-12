"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xcomplex<f32>>
    %4 = "stablehlo.bitcast_convert"(%2) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-2.9432714,-0.240944356), (-1.87811852,2.883020e+00), (2.98283267,-1.20440769)], [(-3.73439598,-0.442267925), (-4.78169394,1.74811637), (-1.23555768,-4.14041948)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-2.9432714,-0.240944356), (-1.87811852,2.883020e+00), (2.98283267,-1.20440769)], [(-3.73439598,-0.442267925), (-4.78169394,1.74811637), (-1.23555768,-4.14041948)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%0) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

