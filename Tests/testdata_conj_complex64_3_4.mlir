"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xcomplex<f32>>
    %4 = "stablehlo.real"(%2) : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %5 = "stablehlo.imag"(%2) : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %6 = "stablehlo.negate"(%5) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    %7 = "stablehlo.complex"(%4, %6) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
    "stablehlo.custom_call"(%7, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-3.92063093,0.947482943), (2.78115916,-2.56036592), (2.53200412,2.17843843), (0.609237313,-2.37602949)], [(-7.69268703,1.68608713), (5.67410326,-2.21368337), (1.93652391,-0.854346096), (-1.19848359,0.0837314874)], [(-0.950235664,-2.71151304), (-0.416245133,-1.51285613), (-1.48842835,0.668041825), (-5.31184292,-1.85015249)]]> : tensor<3x4xcomplex<f32>>}> : () -> tensor<3x4xcomplex<f32>>
    "func.return"(%1) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-3.92063093,-0.947482943), (2.78115916,2.56036592), (2.53200412,-2.17843843), (0.609237313,2.37602949)], [(-7.69268703,-1.68608713), (5.67410326,2.21368337), (1.93652391,0.854346096), (-1.19848359,-0.0837314874)], [(-0.950235664,2.71151304), (-0.416245133,1.51285613), (-1.48842835,-0.668041825), (-5.31184292,1.85015249)]]> : tensor<3x4xcomplex<f32>>}> : () -> tensor<3x4xcomplex<f32>>
    "func.return"(%0) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

