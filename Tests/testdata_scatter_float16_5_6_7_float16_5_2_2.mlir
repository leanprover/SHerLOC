"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      "stablehlo.return"(%arg1) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2x2xi64>, tensor<5x2x2xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xC5C5683E8EBC5144BDBD7446AFBFDBAE97441CBEB6C0203E3539C1B988424BC5C42D50BFE13825452BC459C2F83DDA43B5C0FB3BB144BCC10B3E5CBAEC3E3842034397BD74C40DBE5C4410BB5D3E3DBE12B5EC3ABD3E63BD6EC3EE4493C7724599C8ADBE79C21647BA44CAC49CC1ECC51CB062BEE6C5EBC39BBEDABE4B3E1EBFB2A3A74142C3EDB078B8B543F0B91EC45EC083C281B97044FB3D9DC0E5C380382BBB7844FAB88A3BA4A0BCB847C7544133C0CD423CC4D2C5C0C4594258C3104455448736AB321138154588C111C61DC6214171411A288B3D68B8A2BDDDAF31C132C2B2428C47D8C02441C1C51B4429B4D040EB3916C4E444F0C07FAED24394BA52C4D4BC64BD123E1F3589ADA63860C53CC2CF40CA388AC0BFC105B91CC55BC51CC071406BC465C0DA3FE7C09044A5B0BF423135C6B766C2544175BA6B3C65C45843D6C49041A946B0BA8A44693B3DC2633E9844CB377A43613D6239E13951C3BBC05DC13638C341C2BD51B2CF44B2BE20BF804433C315BE08428AC491BFC44218B807414643AEC371403EC143B47937BBC1AA45A9C4A0C144BB41C0ED2987AD7D44EF45"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-2.144530e+00, 8.935540e-01], [-5.463870e-01, -9.614250e-01]], [[-6.528320e-01, -2.195310e+00], [1.883790e+00, -1.791020e+00]], [[3.716800e+00, 3.580080e+00], [-1.577150e-01, -1.048830e+00]], [[3.759770e+00, -1.056640e+00], [-1.605470e+00, -1.538090e+00]], [[6.039060e+00, 2.016600e-01], [-2.132570e-01, 3.777340e+00]]]> : tensor<5x2x2xf16>}> : () -> tensor<5x2x2xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<5x2x2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xC5C54AC08EBC5144BDBD7446AFBFDBAE9744B1BBB6C0203E3539C1B988424BC5C42D263BE13825452BC459C2F83DDA43B5C0FB3BB144BCC15FB85CBAEC3E3842034397BD74C40DBE5C4410BB5D3E3DBE12B5EC3ABD3E39B96EC3EE4493C7724599C8ADBE79C22ABFBA44CAC49CC1ECC51CB062BEE6C564C09BBEDABE4B3E1EBFB2A3A74142C3EDB078B8B543893F1EC45EC083C281B97044FB3D9DC0E5C380382BBB7844FAB88A3BA4A06F4347C7544133C0CD423CC4D2C5C0C432BC58C3104455448736AB3211381545294311C61DC6214171411A288B3D68B8A2BDDDAF31C10CB1B2428C47D8C02441C1C51B4429B4D040EB3916C4E444F0C07FAED243854352C4D4BC64BD123E1F3589ADA63827BE3CC2CF40CA388AC0BFC105B91CC53ABC1CC071406BC465C0DA3FE7C09044A5B0BF4231356CBE66C2544175BA6B3C65C45843D6C49041A946B0BA8A44693B3DC2633E0A46CB377A43613D6239E13951C3BBC08E433638C341C2BD51B2CF44B2BE20BF743233C315BE08428AC491BFC44218B807414643AEC3D3B23EC143B47937BBC1AA45A9C4A0C144BB41C0ED2987AD7D44EF45"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

