"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xf32>, tensor<3x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xf32>) -> tensor<4x3xf64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf64>) -> tensor<3x6xf64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    "func.return"(%7) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xf32>, tensor<3x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.85355544, -0.751674354, -0.931860387], [-2.97942472, 4.97833872, -3.0331161], [0.376971304, 7.73804426, -1.87362397], [-2.57579374, 2.89119983, 0.675038516]]> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0.092548278060739078, -5.1419313524325911, 4.6704406663416824, 1.091565067637412, 2.327424470260226, 2.1859915239499172], [0.58061786209680044, -3.3449400611298259, 0.85679223097779422, 2.5380628484805761, -3.834596235655483, -4.4763714959094605], [2.2729310577691617, -8.9272968740469167, 1.1533102520043401, -3.3983966760448299, 1.5041182165078604, 0.16756812701518325]]> : tensor<3x6xf64>}> : () -> tensor<3x6xf64>
    "func.return"(%1, %2) : (tensor<4x3xf32>, tensor<3x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.7260333366589293, 20.364154817740829, -10.37567359344779, -0.76424187963146539, -2.8332708253789094, -0.84323292993463006], [-4.279292034061779, 25.745280635347584, -13.147948310531032, 19.690832287602067, -30.586470080666917, -29.306144293689563], [0.26911665571267379, -11.095257430418277, 6.2296485870383504, 26.418448830702673, -31.613055099563343, -34.128264350882233], [2.9746129963855421, -2.4526047598987231, -8.7744054542930332, 2.2323517467427636, -16.06621163245455, -18.459632845487089]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%0) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

