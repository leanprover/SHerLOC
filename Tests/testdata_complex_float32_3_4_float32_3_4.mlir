"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x4xf32>, tensor<3x4xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xcomplex<f32>>
    %5 = "stablehlo.complex"(%3#0, %3#1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x4xf32>, tensor<3x4xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4.13312864, -3.9565289, 3.49465394, 6.253200e-02], [0.410699964, -1.21260703, 2.13978839, -1.51862335], [1.75706387, -0.269945771, -4.01345253, 4.72941542]]> : tensor<3x4xf32>}> : () -> tensor<3x4xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[-4.41265631, -0.327641249, 2.52980947, -0.798980951], [-2.14335227, 3.32583117, 1.07032835, -5.32339954], [-2.90522337, 5.24038172, -4.99515343, -1.49502265]]> : tensor<3x4xf32>}> : () -> tensor<3x4xf32>
    "func.return"(%1, %2) : (tensor<3x4xf32>, tensor<3x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(4.13312864,-4.41265631), (-3.9565289,-0.327641249), (3.49465394,2.52980947), (6.253200e-02,-0.798980951)], [(0.410699964,-2.14335227), (-1.21260703,3.32583117), (2.13978839,1.07032835), (-1.51862335,-5.32339954)], [(1.75706387,-2.90522337), (-0.269945771,5.24038172), (-4.01345253,-4.99515343), (4.72941542,-1.49502265)]]> : tensor<3x4xcomplex<f32>>}> : () -> tensor<3x4xcomplex<f32>>
    "func.return"(%0) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

