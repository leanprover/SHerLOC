"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf32>
    %4 = "stablehlo.real"(%2) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    "func.return"(%4) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-0.132573813,0.224451914), (3.45699096,-1.8876127), (-2.03585815,-4.21114063)], [(-0.65348196,-0.266635329), (0.264353186,-1.13852882), (0.951290428,-2.83588266)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-0.132573813, 3.45699096, -2.03585815], [-0.65348196, 0.264353186, 0.951290428]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%0) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

