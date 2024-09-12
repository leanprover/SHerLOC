"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<1xcomplex<f32>>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xcomplex<f32>>) -> tensor<1xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1xcomplex<f32>>, tensor<1xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<1xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[(1.87978518,-1.79707468), (3.20142984,-3.70751119), (0.516356587,2.9530642)]> : tensor<3xcomplex<f32>>}> : () -> tensor<3xcomplex<f32>>
    "func.return"(%1) : (tensor<3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(3.20142984,-3.70751119)> : tensor<1xcomplex<f32>>}> : () -> tensor<1xcomplex<f32>>
    "func.return"(%0) : (tensor<1xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

