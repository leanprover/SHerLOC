"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui64>, tensor<3x6xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f32>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui64>) -> tensor<4x3xcomplex<f32>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f32>>) -> tensor<3x6xcomplex<f32>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui64>, tensor<3x6xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 0, 0], [0, 2, 4], [1, 1, 1], [0, 4, 0]]> : tensor<4x3xui64>}> : () -> tensor<4x3xui64>
    %2 = "stablehlo.constant"() <{value = dense<[[(2.45522833,-4.48552513), (-0.100219555,7.55077075), (1.84159267,0.578810215), (-2.86075354,1.12263703), (5.32669067,-2.02719879), (0.657366395,-2.13350892)], [(0.543746829,-2.79866958), (0.785029351,0.807303249), (-1.21870899,-3.00168967), (3.06053472,0.790934085), (1.3685559,-1.02736819), (-2.42934918,0.662117362)], [(0.0635634735,-0.945136666), (0.851711571,-3.79603148), (2.30837083,2.51139235), (-0.537010252,2.204260e+00), (0.855342984,0.925947964), (2.66539192,1.88715553)]]> : tensor<3x6xcomplex<f32>>}> : () -> tensor<3x6xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<4x3xui64>, tensor<3x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(2.45522833,-4.48552513), (-0.100219555,7.55077075), (1.84159267,0.578810215), (-2.86075354,1.12263703), (5.32669067,-2.02719879), (0.657366395,-2.13350892)], [(1.34174752,-9.37788581), (4.97690487,-13.569519), (6.79606533,4.042190e+00), (3.97302842,10.3989086), (6.15848351,1.64905548), (5.80286932,8.87285709)], [(3.06253886,-8.22933197), (1.53652143,4.56204271), (2.93125439,0.0885128974), (-0.337229073,4.11783123), (7.55058956,-2.12861896), (0.893409132,0.415763974)], [(2.17498732,-11.1946783), (3.14011741,3.229213), (-4.87483597,-12.0067587), (12.2421389,3.16373634), (5.47422361,-4.10947275), (-9.71739673,2.64846945)]]> : tensor<4x6xcomplex<f32>>}> : () -> tensor<4x6xcomplex<f32>>
    "func.return"(%0) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

