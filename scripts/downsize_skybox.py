#!/usr/bin/env python3

''' Script for downsizing skybox images. '''
import sys
sys.path.append('build')

import os
import math
import cv2
import numpy as np
from multiprocessing import Pool
# from depth_to_skybox import camera_parameters


NUM_WORKER_PROCESSES = 3
DOWNSIZED_WIDTH = 512
DOWNSIZED_HEIGHT = 512

# Constants
SKYBOX_WIDTH = 1024
SKYBOX_HEIGHT = 1024
base_dir = '/disk1/Matterport3D/v1/scans'
skybox_template = '%s/%s/matterport_skybox_images/%s_skybox%d_sami.jpg'
skybox_small_template = '%s/%s/matterport_skybox_images/%s_skybox%d_small.jpg'
skybox_merge_template = '%s/%s/matterport_skybox_images/%s_skybox_small.jpg'

ERROR_FILES = {
  'cV4RVeZvu5T': [
    '36c83d563af44805b4cc578d573c5440',
    'f401018427494857959e417c8563fd12',
    '3d3126b6b7f84116bf9705b9d917a733',
    '8a730c96a3e543b39f0fc734f3a11b85',
    '7c63e07cb6994065bf920c090a8fba72',
    '5c0ffe2f4358463fb6c4da9c7d1099a5',
    '24e37f1b80474123bf2fb14cfca2f6c1',
    '72cb8cdaf2ee4609ba1a895d3463a356',
    'ee3b8f3f42624db9a7bf7ecabbe69739',
    'f0fae4ec3ac248828126a9f12ace354d',
    'cb07db27224a471bb8f006071f5fe18e',
    '0baa0803eaba48cda45a34b1b47a1385',
    'b4371618e0d14af3970088b6ffa6962b'
  ],
  'VzqfbhrpDEA': [
    'f639f546514f48aeaa3e7c4e0a0c07f0',
    'b5acf07292724c28b3266bfebc66e2fe',
    '66aa355521c74f16a0b947ac39a24dc9',
    'c55fb079e8cf43109acb0b7befa87ccf',
    '80bcc6ca11834ce4bd16e431ff726b41',
    'c1df419d993048c58a4502f93bc9af91',
    '9d02ba5f10e9474d82954ec228b796d6',
    'fd1867e4f4f34cb5ac535c26fddf7396',
    'e5e33682d7154b5d940fb29281b07a81',
    '877ed6c601f7420a87c649e7b59a1b94',
    '6522d9139dbf4ffd8e23e6c69b527e70',
    'fd1f297546ea48d686db5992e9419d08',
    '08298d3cccc34b33a9af89ce596abf46',
    '157bed108eb64302a98a20868f4cf421',
    '8063b10b6fb84cd28b5474efb73e3837',
    '4dc461f32fd04fca9f0fce8f953e3a5b',
    '757ece61927642179f02507514fe3461',
    'a74d248fe74c421e9fb1ea25a0f95dee'
  ],
  'D7N2EKCX4Sj': [
    'bc8c5173ef9b4cd48bc4cc36e7dc264b',
    '36fec70092b64b44be632df45ee3ae20',
    'e1f821fbca3e4136a79433b2e52ca1af',
    'ac29866722854dd8a41555cb7c65c2ed',
    '945024ce4509453da087d54e7f7a8ede',
    '36ba7aeeb72e40d69e3e1c775e4bf541',
    '9e7bb9838d0e443c89775617be087d76',
    'a8141dab5c2447a4ba107ddbf43c009f',
    '630f0ee6e8db4d518e31d932d2c5ea4c',
    '10b11d5c0cb7449baca31324b6a371bb',
    'b6bcf908137f4b2d83d66e5fe6c35656',
    '72e7bae027f34e898f90309032ecd3a6',
    '4504315e4d534815bc6f3080f3619cf4',
    '21571a0773434c31bee1950f57be3410',
    '266d088e5b444c7ab90cec89ead1a8a0',
    '90ad5bd4009c43689742a4c6630fe365',
    '37e7b9950ba44d86b27ba9c5145efe85',
    '7a45313ee05048f2bfa610ac81986e1c',
    'bd0a7e319efd4fd98e373a3d2a025566',
    '1619aa4c11c445d092b43bafb4eaefea',
    '61be682aa2134e078e7fd254214ea972',
    'b9a94cf38df045f38b32c07c6fb26376',
    '2ba2fb8b245f40d2866a676ebaf8791b',
    '1bc1667739744f67b2d1bf0566bdb3b2',
    '266e6bf3657149d8a116533e34e6ebf8',
    'fadac209cfc04129998323d048a5c908',
    'bb2ed22637c342ba8acb80a377f234f9',
    'ba3207f81c55467fa7e6760f6232fb07',
    'd923642e556045c793d530b609504b14',
    '585d35a46dd644daad1a135634207b99',
    '53608606748f45c5a69723f062f1baa3',
    '56851a5dbc8c43179c9c74594cd3a554',
    '1cc611304c434e83b14d30f397f0bee3',
    'cf638911e4b74d188c30f79332ed9922',
    '8ae030e5a5504a7eac1269be9eafe6f3',
    '45aa201f03e84e51a94f49c04b60ccf2',
    '216c427533564466bcb81b7d8082dd8c',
    'cd004d728b1b4fd9868416500f7b6cbd',
    '7c0b96985c344cbeac0f7c69e11dc60e',
    '0f3c63abf672414b8e93962fa1a507c0',
    'bde29f48ab814943baf4a7193d143d6e',
    '6f1665f159cf4ba098116113c10f5ee8',
    '8b1537dc0668496f963a3f2ca739a17e',
    'ddd6fc6901314a44aeb414652002249c',
    'e40a12cfb5bc4f4dab071ec7458d378c',
    '21d98b3051984309af072f3fde91c49d',
    '6ac0bff3d8f6441e87513a6ffdc2cd3b',
    'ffd41f034ca44fe280a59b6a192be618',
    'c8c79f5ebf494d4eae51582a5fc08d21',
    'f93b1f2597b6492cb03b4e70763e6a93',
    '2c6afdf7346a4dc3952b97edf85cf144',
    '8d809c69cc10465abc72da3431e8f2e9',
    'c6708fedf6b241148554f8d8592036b6',
    '852a2ee396184da494b42dea29a8b467',
    'b409b06f51b843e3948e146f9e7d7526',
    '60399cb6fd9d42a7a364f155fd152361',
    'babbef7305e34152950788aa9cb363cc',
    'ec1fb8a79e2f4a46a7e758dd00834caf',
    'f12b7ffe7f9e4ece9bd927edf6b52b71',
    '37ff2e51c319423fb5ea58a67b13f388',
    '52474bda7e7043e1be591bea75e3b3ff',
    '77161450d1594012856fe88cc05b5a05',
    '200846287d844664ba5ad352819a8075',
    'e8d145225316479db17c06bd77df5054',
    'f60fb82c0b6341e88ac64e0ac7bb0e5e',
    'd9ed62bd923d4c4fa604ec76b008b631',
    'da25a53c026245a7887e1d584aff79d4',
    'baf515d5319a4892833962977e5548d8',
    'c58491dbb82c46c69c0a85cff3054ca1',
    '099839b0911a409787e237301c90e418',
    '2618e52c54014501b88069f151ae7d5e',
    '74e7d3ed76fe40b8a0d2ea19aaf4196c',
    '84bc1a6b778e45b6beb6b83e07c38ab9',
    'cc95ddcf10124284abb7273dd4759939',
    'dbf9e5bc313145648aa34e10b86ec1bb',
    '2b34f57b3e254473bf1317e43e250c32',
    '03e418be259e4659bb47e1734b02ec67',
    '321971f1bde54fe4ac33def30bdc5921',
    '52131a7696fc47f4bffd1c0cfd57a454',
    '189a4e9b647b4f3196dbf64b02442082',
    '1509b3e5dce94a3a91dda769c5b28ba7',
    '6e1e3e8662774c8b9906d3c0a3b1ed3f',
    '611214ad029a4d4da4b45ebb8a257b3d',
    'ddea830741fd40b6b4f71f82d9b1f68e',
    '9cc6191a6e1642f984e54b1a79b83761',
    '5d9cc77ace3042638282da500e645ca8',
    '377785e3a8ec4ce7bff6de3035f6fcd5',
    '5ee7559f2e4e46cc9a20562c3afb7ce6',
    'ea1d38c4b610463396ec72fdaf5d5ece',
    'd6dacfd7e57645e2a6c614c6402fb168',
    '29da42cfffac43f9a6318ef1c575cd23',
    'f024f440e6c44f7d9771e95b4cdfd370',
    '5e6b957e1751425da6eb622ce0999d34',
    '076a42f6b3e7408cb3a2ee5f2c6d637b',
    '91d591ede2f84f2fbbe0d432feaf22ce',
    '6c160b1ba8564c09ab8491590266f7f1',
    'd2ac4c15536f4a62b285088387a2c9b8',
    '3f2dacac25214cf4951aa1294825354b',
    'd5b80b4e1a3c4a2facd2f1cd2a004b11',
    '5038f4f54ed14bef980e59f9e76ce0de',
    'b551670325194c5daf5a1a230d164734',
    '924f1f75e17748f8a86f11009eaad55e',
    'ae105aae572044048f1e24999e387b8a',
    '5e5ce62aa2f540f1b5978222f8a9deb1',
    '1ad73ac43096496f9b6084071d1d944f',
    '10eb95bf6d0541069c5f084cbb146f11',
    '5e65c1a2439f4594b4a433c805221c05',
    '1aba6e328b4d4211900d8296b88f6127',
    '4e35abb87ff34156956043cc86a6f817',
    'a1890da72aff4056b962636cb8ba8715',
    '23dd7248d68845378c8f4a1d8bdc103e',
    '38622c54054e47b0a9c83623e9b8d91d',
    'c645fc0e7e2a488fbed3580e98bd041e',
    'a962ef167bc24937ae33b17d09bbcb0c',
    '0cb7f98845aa4939a1125d430acb7183',
    '6acea2b64c694b948f50da07d88ed31e',
    '2c30bda8678b4fc4b3918476affbc2ec',
    '4e9e9f69704e4a55a0d48117bb06a343',
    '0026d80fc3c84d91bd088b831d230738',
    'be255b95116f48599c036544957334ec',
    '4187ed8e4a504b7d8caeafa3a0627031',
    'f59c4b4ff65e4a539bbba69e67de001e',
    '5ebc7429c5344395b9849773581b6ebc',
    '4ab6a2ca589d49899f8d7d66151fdfab',
    'ad945086480e44eb9c6cb83c63cd7864',
    'bcf71cd8fa804d5abff84ad22a552450',
    '5dd04cfe127f467488b4828984f6b45b',
    'd4187eee459c477eb49e376eff9fefb4',
    '3384e4644c644c85980470220aa3e101',
    '884e06fb5a9141e8917de14fe8fd8de4',
    'a4843bff16f64daca90b28d0c43c6b00',
    'a1edd25cb2074e288a2a3e1948422c50',
    '7bb04eae8e45423e9fe9e1d078a6924f',
    '3d8ad75bfd9148c29e199b69d68d30d4',
    '75e91f60cc2649228daaf6393b7d10db',
    'e8c8165c6e8b4f19b41d23ebfe3c75ba',
    'e10169b3980a429fa3fc72713215d1b4',
    'cfd9511c06b34cd1b89ba36919a4e6b6',
    'f5bef118db5b41849ce63cbfcefed938',
    '6d88b250b28f4f28afbcf356f6a33909',
    '8db786868c1341709ea8777cb822e063',
    '4ab710f16a804ee7b525322add24f9bb',
    '93848517830e4376a3d3576c16e3685f',
    '238a6b5dc8044fff8d86b5dd1b428074',
    '29a14ad501ee4450970fc128d32582d6',
    'cd7da3a3c0fd4fe69544b18a0b1bea72',
    'c2ff7eb5559440649b93ab262df88a9b',
    '8ff6def5ce0e4ae097ca29d2fdef26f9',
    'ed46f2e3d89d4f2ea2ec7b14affef69d',
    'cec763dd78394e3dac4b14da1611b51a',
    '78ad6e43879c433ea88417b72e61c80f',
    '933b9d8733a84a5bb2d666cba6af204b',
    'ea8c31ffae82450f87d19ca1ac4d01b7',
    '2448bb1de37f4cf6948c822a927feea7',
    '49460faf784e4d93a41569491260bc9c',
    '5b177e47a0d1469695660f79c0c777b4',
    '85f8ca653a7d4d2c8b5415c12bd0abf8',
    'bffa1faac5024b19abf8b4150e634c9c',
    'd9e75c0e52ae4a949957ac8f82f2e3e3',
    '0d89ca4056f64d8d90bf03250ca64eb1',
    '0b4d4c11e012429e8dd6013502094ef3',
    '389f67e9de9b4f729127c2bc2f67a316',
    'faddc614417b4861b4cd05176a35b947',
    'c2e3d3b973294b91bb895469e6fbbb83',
    '005bcfda078e4a79892f5477b347a7fe',
    '9d33a07061f54a5eb58588a5bc9f10da',
    '846adaa1177c459892edfe6f5eadafbc',
    '0f01520129074da5b4c0c58d72442c3a',
    '8e0a04db99fd4c61a9f669d2e6e421a8',
    '96e014c3d5894319b4155277e04ae637',
    '14e9295fc7604291beb548e3f69f54a0',
    '03732a8d7aae49dbaf9b3c25a3fe96d0',
    '975241bb74e744ed97b658a04f189d47',
    '8906c7b18ea149a786d9f2cb83bd2f16',
    '966a5fa6c049457c8ddafb05f6063443',
    '00ef8a0394f447c68796213e64155467',
    '31866025bbc74d7fbf99524d81d35bd1',
    'e164d087432749c6b062e38097661fca',
    'ff9dadef883c42689cabf60157eecbed',
    'dafcbb94981f4b719d2cbb741737830b',
    '8d01d3df74ea481bb82fd00f4ae6bae7',
    '12142264f293430ebc8092e16b47cba0',
    '64884ee49890488f91f1002227e486f4',
    '5787a52b118147cdb26c674f5cb676cb',
    '062df9954974450a9eb914da2727062d',
    '43849d3264fb43a89ab7aeaa0e1adcbb',
    '013aeede69d04db1b99dae6124da4fd5',
    '7292b16bfe1540b8bcc46d523b14b241',
    '9f6684e165434894a5185bb5c3e2c275',
    '994e59a2e5cb469eb29c7ff68c6d49c2',
    'b2c2abdd75de4ea6b203f890bf89cfb0',
    '764cb58a8d31461892ca6bdc675c14df',
    'd0c0af3de31643d79bcc9ebd057ce6e2',
    '1d895bede46e4ce09393f72b7d060631',
    '708553d21c3d4495942f12bf572c1814',
    'eae656fc42b1467089883eae9a69c9da',
    'd193970cbeac4faeae3b9bfee5c35125',
    'ee41ff5a69d9462bb1fe6866d3cba232',
    '40db7e27cdf04e3f9e4569bda65b4bb2',
    '2335251f91a843c3ae05e1fec482438a',
    '5e770c44fdbf44079e0936d2e1aad894',
    '1ba7d881e867480585d80c1222f54028',
    '6ce4614650fd4294852d7fbeb89ef6be',
    '73de1f780b804c92a2441ac92c442c49',
    'd35b945333f5413a9f5586b794cd2680',
    '7371cd65c74040bb9f8aa46f25ceb658',
    '3d11e46400a14c6cbc0ae42a5057ae97',
    '983c7b967c9f43bdbd43508435840af1',
    '7b8e2f039e884764bd1ad6cbc13d9bd0',
    '8368a40efff243108ca481fa695758af',
    'abd113ed091944859eb850ae7cc51595',
    '3776bbf192e7459da28245fdd8712acc',
    '3ca6ed9c10ea4e07bf201598b6e5b8ae',
    'b3e4ec656ec843ea86a932b08131aed9',
    '76869bdceadf4abf9d834fc7c9bbf2f5',
    '8fe8f1504e6c4fc882ae866ad5ab54f2',
    '8b22751d94034268b26b9960c1d28e4e'
  ],
  '8WUmhLawc2A': [
    '897b522c66d94cad979e11a3b93aacfb',
    '6f33ad23fb39423dae9b2b4084dc4f9d',
    '193c6cd5b35c4205b7351026a45a94f9',
    '86e93909e976453aab8745c337fa3587',
    'b3511135d8544de2a630cfae0fd3abc0',
    '6f5fa0f9fc09492cbf907f74294d0530',
    '60cebad6c98a47188700a6866b78914e',
    '050a1991a4a44fc39c4e3d9b1bbcfac4',
    '266a16fdcc73492ba091d2d9f139ca0e',
    'ab4152df16b9405cab8df691483292fd',
    'd97872f39eab43a6bcae7ee773d46949',
    'e2b07d9c7e2242e1b9fbea4900d77d29',
    '01fa989e503048b7979ab2ce51cecf78',
    '87fc0d4b2f7c4fde8e0c6421feb7f21f',
    '06fa9666517c4919ad6c631a828b2b69',
    'd019540341aa43c1adf608df85ca32aa',
    '0c32ae05905d41c68dd632557653bc2f',
    '72da1c859be940f897b0ee33b18658af',
    'b8cf59ada15d499288b90f688bdc8a04',
    '22923b281d2e494498d5e71e25e3fde2'
  ],
  'gTV8FGcVJC9': [
    '5c853a542d8547ab9211296962b4b52e',
    '0edd1e0abdf74bec8991018ad78398f0',
    '74fe9a84209e48c789783e2055ef2b03',
    'df6c3d319a5b4a03a80c8a9439879c42',
    'daf9f19a95ce46779b06b66b16862b8c',
    '847c529729bc4a53a05ffeb603e02524',
    '5dac460c46344eb192a800eb2184e072',
    '251f9a13f8294af5a9b90e6e56cfc6bc',
    'dbede146711e4ab59ab172e1188717df',
    'a4badd6df5b047f8b049643132e20f3b',
    'eaa201e3d25f4dada679230f3982529d',
    'e39dac9fa8e54b5288b0e5393fbe75ed',
    'eea2894644c24b28865c63d877005416',
    '8fe0228447b7438887011330a8a1aba5',
    'a2bb639c306d46eeae8c0cf5b3c112c5',
    'efc735ea55dd4af798246d8cbdb35816',
    '0ad30ece2f9f487d9c77621f8cde489e',
    'fe6da345e69540aa955c507b718d3252',
    '5b09870c4ff74da0b2bab0b32e5b2db2',
    '0661b9a4524144328e4c4dfe6e4e3865',
    'f7ff847c25994bdb82012a3f2452d014',
    'b8e1db15feee4932a15edbfe532e2e54',
    '7f82bb8472f748b9bbf6a9df8de4a882',
    '51e8d5d7eab64cecb73266751f184e93',
    'cb6dab2ac5ae490eacf7011965dc3d9e',
    '0de91705b6224eb08f8af36a2f521e1d',
    '8baff2e8bb77433397463dee4689e6af',
    'd457ca05498a49e7baf4b5c0a086034a',
    '9949ee402d0747bbb878e51bad7c6693',
    'c6081f2842a8486b8a47c1722c6e733c',
    '3413e966bdde4644aa6da940673a2404',
    '4caa00334cee415498e127fd26e99dbb',
    '07ba206e75d2414695f2df4f6a2288a0',
    'cfbdf420a14440efab41907ad13a3341',
    '4efae1e47ceb49c6924518e3715e9ac5',
    'f6d46381319e474c886dbd7eb29c194f',
    'c8522c25453f4c42a396a606e899cf40',
    '407d81168b304008b073f88978ef49fc',
    '6a28b7b3d41b4dc3966612957e7cf5f3',
    '3c1d1711125b41ea8c3dd59f667e2003',
    '83dc82ccd1404ce6b1e5bd1a811be046',
    '064459fcd2e040558f81a52e4e70dd50',
    '893fa86fcc0d4bffbccdf8adea3c04c6',
    'f1132d967a5c4ad4a3359cb38982ef77',
    '7431c442fe124000ac79e985f8b7b7f6',
    '669cfe6fa3ff4e5b8038d264f9be20da',
    'cb45d87383c64a3598ec098e9e4e2f02',
    '478400487cee4213ac2832e1b61a4e97',
    'fd72b5f450374adf9e5b4f91bd7286e1',
    '409f4de350fe48a796c2d6e7fdd62a66',
    '3e86b626cdb841acac271691fa7fad67',
    'e9e01d76089b43fcb5a163f728a0e8fb',
    '08a59d6a927e4fe6999e53031e2f8305',
    '47b2bc8384f442318c797d06bfb8803b',
    '75afde1cc8b94461849d4b0ff9552282',
    '334bcfd77f31438db9f55a90abb6d9c1',
    'bf8b51a64c544a46b6c84e4a37a210dd',
    'b3d98e724a3f4d60beec65126ffb39cc',
    '2f099d8eda624e62be08bf0efa7e0296',
    'ddac6365abe54c89a629db672342a9c8',
    '2a8e4e51f8fb46bba2703e1436a6e03b'
  ]
}

def downsizeWithMerge(scan):
  # Load pano ids
  # intrinsics,_ = camera_parameters(scan)
  # pano_ids = list(set([item.split('_')[0] for item in intrinsics.keys()]))
  # import pdb; pdb.set_trace()
  pano_ids = ERROR_FILES[scan]
  print('Processing scan %s with %d panoramas' % (scan, len(pano_ids)))

  for pano in pano_ids:

    ims = []
    for skybox_ix in range(6):

      # Load and downsize skybox image
      skybox = cv2.imread(skybox_template % (base_dir,scan,pano,skybox_ix))
      ims.append(cv2.resize(skybox,(DOWNSIZED_WIDTH,DOWNSIZED_HEIGHT),interpolation=cv2.INTER_AREA))

    # Save output
    newimg = np.concatenate(ims, axis=1)
    try:
      cv2.imwrite(skybox_merge_template % (base_dir,scan,pano), newimg)
    except Exception as e:
      print(e)


def downsize(scan):

  # Load pano ids
  intrinsics,_ = camera_parameters(scan)
  pano_ids = list(set([item.split('_')[0] for item in intrinsics.keys()]))
  print('Processing scan %s with %d panoramas' % (scan, len(pano_ids)))

  for pano in pano_ids:

    for skybox_ix in range(6):

      # Load and downsize skybox image
      skybox = cv2.imread(skybox_template % (base_dir,scan,pano,skybox_ix))
      newimg = cv2.resize(skybox,(DOWNSIZED_WIDTH,DOWNSIZED_HEIGHT),interpolation=cv2.INTER_AREA)

      # Save output
      try:      
        cv2.imwrite(skybox_small_template % (base_dir,scan,pano,skybox_ix), newimg)
      except Exception as e:
        print(e)


if __name__ == '__main__':

  with open('connectivity/scans.txt') as f:
    # scans = [scan.strip() for scan in f.readlines()]
    scans = [scan for scan in ERROR_FILES.keys()]
    # p = Pool(NUM_WORKER_PROCESSES)
    # p.map(downsizeWithMerge, scans)
    for scan in scans:
      downsizeWithMerge(scan)
